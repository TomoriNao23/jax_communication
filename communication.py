import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh


# =====================================
# HaloExchange 类
# =====================================
class HaloExchange:
    """
    封装 cubed-sphere 拓扑的 halo 交换逻辑。

    物理边数量：12（每个面 4 条边，共 24 面-边，两两配对 = 12 条物理边）。
    通信方案：12 条边用图着色分为 6 轮，每轮 1 次 lax.ppermute（双射置换）。

    Args:
        halo      : halo 宽度
        nx_local  : 单 tile 本地 x 方向网格点数（不含 halo）
        ny_local  : 单 tile 本地 y 方向网格点数（不含 halo）
        mesh      : JAX Mesh，需包含 'tile' 轴
    """

    E, W, N, S = 0, 1, 2, 3
    ID = 0
    TRANS = 1
    FLIP_I_TRANS = 2
    FLIP_J_TRANS = 3

    # 12 条物理边（每条只列一次，均为自逆变换）
    # (tile_A, dir_A, tile_B, dir_B, transform)
    # tile_A 的 dir_A 数据 -> tile_B 的 dir_B halo（施加 transform）
    # tile_B 的 dir_B 数据 -> tile_A 的 dir_A halo（施加同一 transform，因为均自逆）
    # 面布局         [4]
    #           [3] [0] [1] [2]
    #               [5]
    EDGES = [
        # 赤道带东西环绕（4 条，ID 变换）
        (0, E, 1, W, ID),
        (1, E, 2, W, ID),
        (2, E, 3, W, ID),
        (3, E, 0, W, ID),
        # 赤道面与北极面 4（旋转 0°/90°/180°/270°）
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

    # ---------------------------------------------
    # 类属性：全局配置 + 预编译对象
    # ---------------------------------------------
    halo: int = 0
    nx_local: int = 0
    ny_local: int = 0
    mesh: Mesh | None = None
    ntile: int = 6
    _n_pad: int = 0
    _rounds = None
    _exchange_jit = None

    @classmethod
    def configure(
        cls,
        halo: int,
        nx_local: int,
        ny_local: int,
        mesh: Mesh,
    ) -> None:
        """
        完成 HaloExchange 的全局配置，相当于原来的 __init__。

        之后通过 `HaloExchange.exchange(u)` 直接调用，无需实例化对象。
        """
        cls.halo = halo
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls.mesh = mesh
        cls._n_pad = max(nx_local, ny_local)  # 统一 padding 长度

        # 图着色，得到 [(edges_in_round, perm), ...]
        cls._rounds = cls._color_edges()

        # 预编译 vmap+jit，避免重复编译
        cls._exchange_jit = jax.jit(
            jax.vmap(cls._exchange_single, axis_name='tile')
        )

    # ------------------------------------------------------------------
    # 图着色：将 12 条边分为最少轮次（每轮内的边不共享 tile）
    # ------------------------------------------------------------------
    @classmethod
    def _color_edges(cls):
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
            # 构造双射置换 perm：参与边的两端互换，其余 tile 恒等
            swapped = set()
            for (tA, _, tB, _, _) in group:
                swapped.add(tA)
                swapped.add(tB)
            perm = []
            for (tA, _, tB, _, _) in group:
                perm += [(tA, tB), (tB, tA)]
            perm += [(t, t) for t in range(cls.ntile) if t not in swapped]
            # 验证置换合法性
            assert sorted(p[0] for p in perm) == list(range(cls.ntile)), \
                f"轮{c} src 不完整: {perm}"
            assert sorted(p[1] for p in perm) == list(range(cls.ntile)), \
                f"轮{c} dst 不完整: {perm}"
            rounds.append((group, perm))
        return rounds

    # ------------------------------------------------------------------
    # 打包：取 direction 方向的有效数据，padding 到 (h, n_pad)
    # 统一缓冲区形状为 (h, n_pad)，避免跨边形状不一致
    # ------------------------------------------------------------------
    @classmethod
    def _pack_edge(cls, u: jnp.ndarray, direction: int) -> jnp.ndarray:
        h, n = cls.halo, cls._n_pad
        if direction == cls.E:
            raw = u[h:-h, -2*h:-h].T   # (ny_local, h) -> (h, ny_local)
        elif direction == cls.W:
            raw = u[h:-h,   h:2*h].T   # (h, ny_local)
        elif direction == cls.N:
            raw = u[-2*h:-h, h:-h]     # (h, nx_local)
        else:  # S
            raw = u[  h:2*h, h:-h]     # (h, nx_local)
        cur = raw.shape[1]
        if cur < n:
            raw = jnp.concatenate(
                [raw, jnp.zeros((h, n - cur), dtype=raw.dtype)], axis=1
            )
        return raw  # (h, n_pad)

    # ------------------------------------------------------------------
    # 解包并写入 halo：静态分支（Python-level dispatch），
    # 完全避免 lax.switch/lax.cond 的输出形状约束。
    #
    # data     : (h, n_pad)，ppermute 收到的原始缓冲
    # recv_dir : 接收方要写入的 halo 方向（Python int，编译期常量）
    # tid      : 变换 id（Python int，编译期常量）
    # 返回值   : 更新后的 u
    # ------------------------------------------------------------------
    @classmethod
    def _unpack_and_write(
        cls,
        u: jnp.ndarray,
        data: jnp.ndarray,
        recv_dir: int,
        tid: int,
    ) -> jnp.ndarray:
        h = cls.halo

        # 1. 截取有效长度（去 padding）
        n = cls.ny_local if recv_dir in (cls.E, cls.W) else cls.nx_local
        raw = data[:, :n]   # (h, n)

        # 2. 施加变换（Python 静态分支，编译期展开，无形状约束问题）
        #    变换后形状：
        #      ID          -> (h, n)
        #      TRANS       -> (n, h)   （转置）
        #      FLIP_I_TRANS-> (h, n)   （列翻转）
        #      FLIP_J_TRANS-> (h, n)   （行翻转）
        if tid == cls.ID:
            t = raw
        elif tid == cls.TRANS:
            t = raw.T          # (n, h)
        elif tid == cls.FLIP_I_TRANS:
            t = raw[:, ::-1]   # (h, n)
        else:  # FLIP_J_TRANS
            t = raw[::-1, :]   # (h, n)

        # 3. 写入 halo（形状与目标区域匹配）
        #    E/W halo 目标形状 (ny_local, h)；N/S halo 目标形状 (h, nx_local)
        #    TRANS 后 t 形状已经是目标形状（(n,h) 对 E/W，(n,h) 对 N/S）；
        #    其余变换 t 形状为 (h,n)，需要对 E/W 转置。
        if recv_dir == cls.E:
            # 目标 (ny_local, h)：t 是 (n,h) [TRANS] 或 (h,n) [其余需 .T]
            arr = t if tid == cls.TRANS else t.T
            return u.at[h:-h, -h:].set(arr)
        elif recv_dir == cls.W:
            arr = t if tid == cls.TRANS else t.T
            return u.at[h:-h, :h].set(arr)
        elif recv_dir == cls.N:
            # 目标 (h, nx_local)：t 是 (n,h) [TRANS 需 .T] 或 (h,n) [其余直接用]
            arr = t.T if tid == cls.TRANS else t
            return u.at[-h:, h:-h].set(arr)
        else:  # S
            arr = t.T if tid == cls.TRANS else t
            return u.at[:h, h:-h].set(arr)

    # ------------------------------------------------------------------
    # 单轮交换（供 vmap 使用）
    #
    # tA/tB/dA/dB/tid 均为 Python 编译期常量，所有分支静态展开，
    # 不使用 lax.switch / lax.cond，彻底规避输出形状约束问题。
    # ------------------------------------------------------------------
    @classmethod
    def _one_round(
        cls,
        u: jnp.ndarray,
        edges_in_round: list,
        perm: list,
    ) -> jnp.ndarray:
        tile_id = lax.axis_index('tile')
        h, n_pad = cls.halo, cls._n_pad
        dummy = jnp.zeros((h, n_pad), dtype=u.dtype)

        # ---- 打包发送缓冲（jnp.where 按 tile_id 选择，形状统一） ----
        send_data = dummy
        for (tA, dA, tB, dB, _) in edges_in_round:
            send_data = jnp.where(tile_id == tA, cls._pack_edge(u, dA), send_data)
            send_data = jnp.where(tile_id == tB, cls._pack_edge(u, dB), send_data)

        # ---- 通信：双射置换 ----
        recv_data = lax.ppermute(send_data, axis_name='tile', perm=perm)

        # ---- 写入 halo（静态展开，每条边对两个 tile 各写一次）----
        # tile_A 收到的是 tile_B 的 dB 数据 -> 施加变换 tid -> 写入 dA halo
        # tile_B 收到的是 tile_A 的 dA 数据 -> 施加变换 tid -> 写入 dB halo
        # jnp.where 在两个完整数组间选择，编译器会消除不选中的那个分支的副作用。
        for (tA, dA, tB, dB, tid) in edges_in_round:
            u_if_A = cls._unpack_and_write(u, recv_data, dA, tid)
            u_if_B = cls._unpack_and_write(u, recv_data, dB, tid)
            # 先应用 tile_A 的更新
            u = jnp.where(tile_id == tA, u_if_A, u)
            # 再应用 tile_B 的更新（tile_A ≠ tile_B，互不干扰）
            u = jnp.where(tile_id == tB, u_if_B, u)

        return u

    # ------------------------------------------------------------------
    # 单 tile 完整交换（供 vmap 使用）
    # ------------------------------------------------------------------
    @classmethod
    def _exchange_single(cls, u_local: jnp.ndarray) -> jnp.ndarray:
        for edges_in_round, perm in cls._rounds:
            u_local = cls._one_round(u_local, edges_in_round, perm)
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
