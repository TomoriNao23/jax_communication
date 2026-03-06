import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh


class HaloExchange:

    E, W, N, S = 0, 1, 2, 3

    # 简化的 Transform 类别
    ID = 0
    FLIP_I_TRANS = 1

    ntile = 6

    halo = 0
    nx_local = 0
    ny_local = 0
    _n_pad = 0

    mesh = None
    EDGES = None
    _rounds = None
    _schedule = None
    
    _exchange_jit = None
    _boundary_exchange_jit = None

    # -------------------------------------------------
    # C网格特有的物理棱线通信指标配置
    # -------------------------------------------------
    IDX_W = 3
    IDX_E = -3
    IDX_S = 3
    IDX_N = -3

    # -------------------------------------------------
    # transform 判断
    # -------------------------------------------------

    @classmethod
    def _infer_transform(cls, dA, dB):
        table = {
            (cls.E, cls.W): cls.ID,
            (cls.W, cls.E): cls.ID,
            (cls.N, cls.S): cls.ID,
            (cls.S, cls.N): cls.ID,

            (cls.E, cls.S): cls.FLIP_I_TRANS,
            (cls.W, cls.N): cls.FLIP_I_TRANS,
            (cls.S, cls.E): cls.FLIP_I_TRANS,
            (cls.N, cls.W): cls.FLIP_I_TRANS,
        }
        return table[(dA, dB)]

    # -------------------------------------------------
    # 生成拓扑
    # -------------------------------------------------

    @classmethod
    def _generate_edges(cls):
        edges = []
        for tile in range(cls.ntile):
            if tile % 2 == 0: 
                rules = {
                    cls.S: ((tile + 4) % 6, cls.E),
                    cls.E: ((tile + 2) % 6, cls.S),
                    cls.N: ((tile + 1) % 6, cls.S),
                    cls.W: ((tile + 5) % 6, cls.E),
                }
            else: 
                rules = {
                    cls.E: ((tile + 1) % 6, cls.W),
                    cls.N: ((tile + 2) % 6, cls.W),
                    cls.W: ((tile + 4) % 6, cls.N),
                    cls.S: ((tile + 5) % 6, cls.N),
                }

            for dA, (nb, dB) in rules.items():
                if tile < nb:
                    tid = cls._infer_transform(dA, dB)
                    edges.append((tile, dA, nb, dB, tid))
        return edges

    # -------------------------------------------------
    # configure
    # -------------------------------------------------

    @classmethod
    def configure(cls, halo, nx_local, ny_local, mesh):
        cls.halo = halo
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls.mesh = mesh
        cls._n_pad = max(nx_local, ny_local)

        cls.EDGES = cls._generate_edges()
        cls._rounds = cls._color_edges()
        cls._schedule = cls._build_schedule()

        cls._exchange_jit = jax.jit(
            jax.vmap(cls._exchange_single, axis_name="tile")
        )
        
        cls._boundary_exchange_jit = jax.jit(
            jax.vmap(cls._boundary_exchange_single, axis_name="tile")
        )

    # -------------------------------------------------
    # 图着色
    # -------------------------------------------------

    @classmethod
    def _color_edges(cls):
        n = len(cls.EDGES)
        conflicts = [[False]*n for _ in range(n)]

        for i, (a, _, b, _, _) in enumerate(cls.EDGES):
            for j, (c, _, d, _, _) in enumerate(cls.EDGES):
                if i != j and {a, b} & {c, d}:
                    conflicts[i][j] = True

        colors = [-1]*n
        for i in range(n):
            used = {colors[j] for j in range(n) if conflicts[i][j] and colors[j] >= 0}
            c = 0
            while c in used:
                c += 1
            colors[i] = c

        rounds = []
        for c in range(max(colors) + 1):
            group = [cls.EDGES[i] for i in range(n) if colors[i] == c]
            swapped = set()
            for a, _, b, _, _ in group:
                swapped.add(a)
                swapped.add(b)

            perm = []
            for a, _, b, _, _ in group:
                perm.append((a, b))
                perm.append((b, a))

            for t in range(cls.ntile):
                if t not in swapped:
                    perm.append((t, t))
            rounds.append((group, perm))

        return rounds

    # -------------------------------------------------

    @classmethod
    def _build_schedule(cls):
        schedule = []
        for group, _ in cls._rounds:
            round_sched = [None]*cls.ntile
            for tA, dA, tB, dB, tid in group:
                round_sched[tA] = (dA, tB, tid)
                round_sched[tB] = (dB, tA, tid)
            schedule.append(round_sched)
        return schedule

    # -------------------------------------------------
    # 原有的 Halo Exchange 通信逻辑 (保持不变)
    # -------------------------------------------------

    @classmethod
    def _pack_edge(cls, u, d):
        h = cls.halo
        n = cls._n_pad

        if d == cls.E:
            raw = u[h:-h, -2*h:-h].T
        elif d == cls.W:
            raw = u[h:-h, h:2*h].T
        elif d == cls.N:
            raw = u[-2*h:-h, h:-h]
        else:
            raw = u[h:2*h, h:-h]

        cur = raw.shape[1]
        if cur < n:
            raw = jnp.concatenate(
                [raw, jnp.zeros((h, n - cur), dtype=raw.dtype)], axis=1
            )
        return raw

    @classmethod
    def _apply_transform(cls, data, recv_dir, tid):
        if recv_dir in (cls.E, cls.W):
            n = cls.ny_local
        else:
            n = cls.nx_local

        raw = data[:, :n]

        if tid == cls.ID:
            t = raw
        elif tid == cls.FLIP_I_TRANS:
            t = raw.T[::-1, :]
        else:
            raise ValueError("invalid transform")

        if recv_dir in (cls.E, cls.W):
            if t.shape[0] != cls.ny_local:
                t = t.T
        else:
            if t.shape[0] != cls.halo:
                t = t.T
        return t

    @classmethod
    def _one_round(cls, u, round_idx, perm):
        tile_id = lax.axis_index("tile")
        h = cls.halo
        n = cls._n_pad
        sched = cls._schedule[round_idx]

        send = jnp.zeros((h, n), dtype=u.dtype)

        for t in range(cls.ntile):
            entry = sched[t]
            if entry is None:
                continue
            d, _, _ = entry
            
            send = lax.cond(
                tile_id == t,
                lambda _: cls._pack_edge(u, d),
                lambda _: send,
                None
            )

        recv = lax.ppermute(send, "tile", perm)

        for t in range(cls.ntile):
            entry = sched[t]
            if entry is None:
                continue
            d, _, tid = entry
            arr = cls._apply_transform(recv, d, tid)

            def write(_):
                if d == cls.E:
                    return u.at[h:-h, -h:].set(arr)
                elif d == cls.W:
                    return u.at[h:-h, :h].set(arr)
                elif d == cls.N:
                    return u.at[-h:, h:-h].set(arr)
                else:
                    return u.at[:h, h:-h].set(arr)

            u = lax.cond(tile_id == t, write, lambda _: u, None)

        return u

    @classmethod
    def _exchange_single(cls, u):
        for r, (_, perm) in enumerate(cls._rounds):
            u = cls._one_round(u, r, perm)
        return u

    @classmethod
    def exchange(cls, u):
        with cls.mesh:
            return cls._exchange_jit(u)

    # -------------------------------------------------
    # C 网格边界棱线均值同步逻辑
    # -------------------------------------------------

    @classmethod
    def _pack_boundary_edge(cls, u, v, d):
        h = cls.halo
        n = cls._n_pad

        if d == cls.E:
            line = v[h:-h, cls.IDX_E]
        elif d == cls.W:
            line = v[h:-h, cls.IDX_W]
        elif d == cls.N:
            line = u[cls.IDX_N, h:-h]
        else: # S
            line = u[cls.IDX_S, h:-h]

        raw = jnp.zeros((h, len(line)), dtype=u.dtype)
        raw = raw.at[0, :].set(line)

        cur = raw.shape[1]
        if cur < n:
            raw = jnp.concatenate(
                [raw, jnp.zeros((h, n - cur), dtype=raw.dtype)], axis=1
            )
        return raw

    @classmethod
    def _boundary_one_round(cls, u, v, round_idx, perm):
        tile_id = lax.axis_index("tile")
        h = cls.halo
        n = cls._n_pad
        sched = cls._schedule[round_idx]

        send = jnp.zeros((h, n), dtype=u.dtype)

        for t in range(cls.ntile):
            entry = sched[t]
            if entry is None:
                continue
            d, _, _ = entry
            
            send = lax.cond(
                tile_id == t,
                lambda _: cls._pack_boundary_edge(u, v, d),
                lambda _: send,
                None
            )

        recv = lax.ppermute(send, "tile", perm)

        for t in range(cls.ntile):
            entry = sched[t]
            if entry is None:
                continue
            d, _, tid = entry

            arr = cls._apply_transform(recv, d, tid)

            def write(_):
                # 【核心修正】：严格保留原版的正确提取轴
                # 同步调换更新变量以修正你观察到的“本该 u 变却变成 v 变”的错位现象
                if d == cls.E:
                    recv_line = arr[:, 0]
                    curr_line = v[h:-h, cls.IDX_E]
                    val = ((curr_line + recv_line) / 2.0).astype(v.dtype)
                    return u, v.at[h:-h, cls.IDX_E].set(val)
                elif d == cls.W:
                    recv_line = arr[:, 0]
                    curr_line = v[h:-h, cls.IDX_W]
                    val = ((curr_line + recv_line) / 2.0).astype(v.dtype)
                    return u, v.at[h:-h, cls.IDX_W].set(val)
                elif d == cls.N:
                    recv_line = arr[0, :]
                    curr_line = u[cls.IDX_N, h:-h]
                    val = ((curr_line + recv_line) / 2.0).astype(u.dtype)
                    return u.at[cls.IDX_N, h:-h].set(val), v
                else: # S
                    recv_line = arr[0, :]
                    curr_line = u[cls.IDX_S, h:-h]
                    val = ((curr_line + recv_line) / 2.0).astype(u.dtype)
                    return u.at[cls.IDX_S, h:-h].set(val), v

            u, v = lax.cond(tile_id == t, write, lambda _: (u, v), None)

        return u, v

    @classmethod
    def _boundary_exchange_single(cls, u, v):
        for r, (_, perm) in enumerate(cls._rounds):
            u, v = cls._boundary_one_round(u, v, r, perm)
        return u, v

    @classmethod
    def boundary_communication(cls, u, v):
        with cls.mesh:
            return cls._boundary_exchange_jit(u, v)
    
    # -------------------------------------------------
    # 打印通信计划
    # -------------------------------------------------

    @classmethod
    def print_schedule_info(cls):
        if cls._rounds is None:
            print("错误: HaloExchange 尚未配置。请先调用 configure()。")
            return

        print(f"通信轮次数: {len(cls._rounds)}")
        dn = {cls.E: 'E', cls.W: 'W', cls.N: 'N', cls.S: 'S'}
        
        for i, (edges, _) in enumerate(cls._rounds):
            round_info = [
                (e[0] + 1, dn[e[1]], e[2] + 1, dn[e[3]]) 
                for e in edges
            ]
            print(f"  轮{i}: {round_info}")