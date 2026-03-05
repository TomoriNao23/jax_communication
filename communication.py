import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh


class HaloExchange:
    """
    cubed-sphere 拓扑的 halo 交换。

    面布局         [4]
              [3] [0] [1] [2]
                  [5]

    物理边 12 条，图着色分 6 轮，每轮 1 次 lax.ppermute。

    数据结构
    --------
    EDGES : list[tuple]
        12 条物理边，每条格式为 (tile_A, dir_A, tile_B, dir_B, transform)。
        变换均自逆，tile_A/B 可对称使用。

    _rounds : list[tuple]
        图着色结果，每个元素为 (edges_in_round, perm)。
        - edges_in_round : 本轮参与通信的边列表
        - perm           : lax.ppermute 所需的双射置换列表 [(src, dst), ...]

    _schedule : list[list[tuple | None]]
        路由调度表，_schedule[round_idx][tile_id] 为该 tile 在该轮的路由。
        - (direction, partner, tid) : 发送/接收方向、通信对端、变换 id
        - None                      : 本轮不参与通信（IDLE）
        direction 同时用作 send_dir 和 write_halo_dir（两者对任意边恒相等）。
    """

    # 方向常量
    E, W, N, S = 0, 1, 2, 3

    # 变换 id
    ID           = 0   # 恒等
    TRANS        = 1   # 转置
    FLIP_I_TRANS = 2   # 列翻转
    FLIP_J_TRANS = 3   # 行翻转

    EDGES = [
        # 赤道带东西环绕
        (0, E, 1, W, ID),
        (1, E, 2, W, ID),
        (2, E, 3, W, ID),
        (3, E, 0, W, ID),
        # 赤道面与北极面 4
        (0, N, 4, S, ID),
        (1, N, 4, W, TRANS),
        (2, N, 4, N, FLIP_I_TRANS),
        (3, N, 4, E, FLIP_J_TRANS),
        # 赤道面与南极面 5
        (0, S, 5, N, ID),
        (1, S, 5, E, TRANS),
        (2, S, 5, S, FLIP_I_TRANS),
        (3, S, 5, W, FLIP_J_TRANS),
    ]

    halo: int         = 0
    nx_local: int     = 0
    ny_local: int     = 0
    ntile: int        = 6
    _n_pad: int       = 0
    mesh: Mesh | None = None
    _rounds           = None
    _schedule         = None
    _exchange_jit     = None

    # ------------------------------------------------------------------
    # 配置入口
    # ------------------------------------------------------------------
    @classmethod
    def configure(
        cls,
        halo: int,
        nx_local: int,
        ny_local: int,
        mesh: Mesh,
    ) -> None:
        cls.halo      = halo
        cls.nx_local  = nx_local
        cls.ny_local  = ny_local
        cls.mesh      = mesh
        cls._n_pad    = max(nx_local, ny_local)

        cls._rounds   = cls._color_edges()
        cls._schedule = cls._build_schedule()

        cls._exchange_jit = jax.jit(
            jax.vmap(cls._exchange_single, axis_name='tile')
        )

    # ------------------------------------------------------------------
    # 图着色 → _rounds
    # ------------------------------------------------------------------
    @classmethod
    def _color_edges(cls) -> list:
        n = len(cls.EDGES)
        conflicts = [[False] * n for _ in range(n)]
        for i, (tA1, _, tB1, _, _) in enumerate(cls.EDGES):
            for j, (tA2, _, tB2, _, _) in enumerate(cls.EDGES):
                if i != j and {tA1, tB1} & {tA2, tB2}:
                    conflicts[i][j] = True

        colors = [-1] * n
        for i in range(n):
            used = {colors[j] for j in range(n) if conflicts[i][j] and colors[j] >= 0}
            colors[i] = next(c for c in range(n) if c not in used)

        rounds = []
        for c in range(max(colors) + 1):
            group = [cls.EDGES[i] for i in range(n) if colors[i] == c]
            swapped = set()
            for tA, _, tB, _, _ in group:
                swapped.add(tA)
                swapped.add(tB)
            perm = []
            for tA, _, tB, _, _ in group:
                perm += [(tA, tB), (tB, tA)]
            perm += [(t, t) for t in range(cls.ntile) if t not in swapped]
            assert sorted(p[0] for p in perm) == list(range(cls.ntile))
            assert sorted(p[1] for p in perm) == list(range(cls.ntile))
            rounds.append((group, perm))
        return rounds

    # ------------------------------------------------------------------
    # 路由表 → _schedule
    # ------------------------------------------------------------------
    @classmethod
    def _build_schedule(cls) -> list:
        schedule = []
        for group, _ in cls._rounds:
            round_sched: list = [None] * cls.ntile
            for tA, dA, tB, dB, tid in group:
                assert round_sched[tA] is None, f"tile {tA} 在同一轮内被分配了两条边"
                assert round_sched[tB] is None, f"tile {tB} 在同一轮内被分配了两条边"
                round_sched[tA] = (dA, tB, tid)
                round_sched[tB] = (dB, tA, tid)
            schedule.append(round_sched)
        return schedule

    # ------------------------------------------------------------------
    # 打包：取 direction 方向的有效数据，padding 到 (h, n_pad)
    # ------------------------------------------------------------------
    @classmethod
    def _pack_edge(cls, u: jnp.ndarray, direction: int) -> jnp.ndarray:
        h, n = cls.halo, cls._n_pad
        if direction == cls.E:
            raw = u[h:-h, -2*h:-h].T
        elif direction == cls.W:
            raw = u[h:-h,   h:2*h].T
        elif direction == cls.N:
            raw = u[-2*h:-h, h:-h]
        else:  # S
            raw = u[  h:2*h, h:-h]
        cur = raw.shape[1]
        if cur < n:
            raw = jnp.concatenate(
                [raw, jnp.zeros((h, n - cur), dtype=raw.dtype)], axis=1
            )
        return raw  # (h, n_pad)

    # ------------------------------------------------------------------
    # 变换：(h, n_pad) 缓冲 → 目标 halo 形状
    #   E/W halo 目标形状 (ny_local, h)
    #   N/S halo 目标形状 (h, nx_local)
    # ------------------------------------------------------------------
    @classmethod
    def _apply_transform(
        cls,
        data: jnp.ndarray,
        recv_dir: int,
        tid: int,
    ) -> jnp.ndarray:
        n = cls.ny_local if recv_dir in (cls.E, cls.W) else cls.nx_local
        raw = data[:, :n]

        if tid == cls.ID:
            t = raw
        elif tid == cls.TRANS:
            t = raw.T
        elif tid == cls.FLIP_I_TRANS:
            t = raw[:, ::-1]
        else:  # FLIP_J_TRANS
            t = raw[::-1, :]

        if recv_dir in (cls.E, cls.W):
            return t if tid == cls.TRANS else t.T
        else:
            return t.T if tid == cls.TRANS else t

    # ------------------------------------------------------------------
    # 写入 halo：lax.cond 只在 target_tile 上执行 .at[].set()
    # ------------------------------------------------------------------
    @classmethod
    def _write_halo_if_tile(
        cls,
        u: jnp.ndarray,
        recv_data: jnp.ndarray,
        recv_dir: int,
        tid: int,
        target_tile: int,
        tile_id: jnp.ndarray,
    ) -> jnp.ndarray:
        h   = cls.halo
        arr = cls._apply_transform(recv_data, recv_dir, tid)

        if recv_dir == cls.E:
            true_fn = lambda _: u.at[h:-h, -h:].set(arr)
        elif recv_dir == cls.W:
            true_fn = lambda _: u.at[h:-h, :h].set(arr)
        elif recv_dir == cls.N:
            true_fn = lambda _: u.at[-h:, h:-h].set(arr)
        else:  # S
            true_fn = lambda _: u.at[:h, h:-h].set(arr)

        return lax.cond(tile_id == target_tile, true_fn, lambda _: u, None)

    # ------------------------------------------------------------------
    # 单轮交换：按路由表展开，每 tile 只生成 1 次 _pack_edge
    # ------------------------------------------------------------------
    @classmethod
    def _one_round(
        cls,
        u: jnp.ndarray,
        round_idx: int,
        perm: list,
    ) -> jnp.ndarray:
        tile_id     = lax.axis_index('tile')
        h, n_pad    = cls.halo, cls._n_pad
        round_sched = cls._schedule[round_idx]

        # 打包：路由表按 tile 展开，_pack_edge 在 lambda 内部
        send_data = jnp.zeros((h, n_pad), dtype=u.dtype)
        for t in range(cls.ntile):
            entry = round_sched[t]
            if entry is None:
                send_data = lax.cond(
                    tile_id == t,
                    lambda _: jnp.zeros((h, n_pad), dtype=u.dtype),
                    lambda _, s=send_data: s,
                    None,
                )
            else:
                direction, _, _ = entry
                send_data = lax.cond(
                    tile_id == t,
                    lambda _, d=direction: cls._pack_edge(u, d),
                    lambda _, s=send_data: s,
                    None,
                )

        # 通信
        recv_data = lax.ppermute(send_data, axis_name='tile', perm=perm)

        # 写入 halo：IDLE tile 完全跳过
        for t in range(cls.ntile):
            entry = round_sched[t]
            if entry is None:
                continue
            direction, _, tid = entry
            u = cls._write_halo_if_tile(u, recv_data, direction, tid, t, tile_id)

        return u

    # ------------------------------------------------------------------
    # 单 tile 完整交换
    # ------------------------------------------------------------------
    @classmethod
    def _exchange_single(cls, u_local: jnp.ndarray) -> jnp.ndarray:
        for round_idx, (_, perm) in enumerate(cls._rounds):
            u_local = cls._one_round(u_local, round_idx, perm)
        return u_local

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    @classmethod
    def exchange(cls, u: jnp.ndarray) -> jnp.ndarray:
        """
        执行全部 tile 的 halo 交换。

        Args:
            u: shape = (ntile, nx_local + 2*halo, ny_local + 2*halo)

        Returns:
            halo 区域已填充的数组，shape 不变。
        """
        if cls._exchange_jit is None or cls.mesh is None:
            raise RuntimeError("HaloExchange 未配置，请先调用 HaloExchange.configure(...)")
        with cls.mesh:
            return cls._exchange_jit(u)