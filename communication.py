import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh

# ==========================================
# 1. 拓扑类 (Topology)
# ==========================================
class Topology:
    """
    负责处理网格的拓扑连接规则，内置方向和旋转常量配置。
    """
    # 方向常量
    E, W, N, S = 0, 1, 2, 3
    # 变换规则常量
    ID = 0
    FLIP_I_TRANS = 1
    
    NTILE = 6

    @classmethod
    def get_transform(cls, dir_a, dir_b):
        """判断两个相邻面拼接时的数据转换规则"""
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
        return table[(dir_a, dir_b)]

    @classmethod
    def generate_edges(cls):
        """生成拓扑边：记录面上相互连接的几何规则"""
        edges = []
        for tile in range(cls.NTILE):
            # 抽象奇偶面相邻规则
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

            # 生成 nsew 旋转规则及拓扑连接边
            for d_a, (nb, d_b) in rules.items():
                if tile < nb:
                    tid = cls.get_transform(d_a, d_b)
                    edges.append((tile, d_a, nb, d_b, tid))
        return edges


# ==========================================
# 2. 贪心图着色类 (GreedyColoring)
# ==========================================
class GreedyColoring:
    """处理通信依赖与冲突的调度器"""
    
    @staticmethod
    def color_edges(edges, ntile):
        """将边分配到不冲突的轮次中"""
        n = len(edges)
        conflicts = [[False] * n for _ in range(n)]

        for i, (a, _, b, _, _) in enumerate(edges):
            for j, (c, _, d, _, _) in enumerate(edges):
                if i != j and {a, b} & {c, d}:
                    conflicts[i][j] = True

        colors = [-1] * n
        for i in range(n):
            used_colors = {colors[j] for j in range(n) if conflicts[i][j] and colors[j] >= 0}
            c = 0
            while c in used_colors:
                c += 1
            colors[i] = c

        rounds = []
        for c in range(max(colors) + 1):
            group = [edges[i] for i in range(n) if colors[i] == c]
            swapped = {t for edge in group for t in (edge[0], edge[2])}

            perm = [(a, b) for a, _, b, _, _ in group] + [(b, a) for a, _, b, _, _ in group]
            
            # 未参与交换的 tile 指向自己
            perm.extend([(t, t) for t in range(ntile) if t not in swapped])
            rounds.append((group, perm))

        return rounds

    @staticmethod
    def build_schedule(rounds, ntile):
        """生成并格式化每轮每个 tile 的详细收发计划表"""
        schedule = []
        for group, _ in rounds:
            round_sched = [None] * ntile
            for tA, dA, tB, dB, tid in group:
                round_sched[tA] = (dA, tB, tid)
                round_sched[tB] = (dB, tA, tid)
            schedule.append(round_sched)
        return schedule


# ==========================================
# 3. 通信类 (Communication)
# ==========================================
class Communication:
    """负责处理域内 Update 与边界均值同步的通信核心"""
    
    # C网格特有的物理棱线通信指标配置
    IDX_W = 3
    IDX_E = -3
    IDX_S = 3
    IDX_N = -3

    halo = 0
    nx_local = 0
    ny_local = 0
    _n_pad = 0
    mesh = None

    _rounds = None
    _schedule = None

    _update_domain_jit = None
    _boundary_communication_jit = None

    @classmethod
    def configure(cls, halo, nx_local, ny_local, mesh):
        cls.halo = halo
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls.mesh = mesh
        cls._n_pad = max(nx_local, ny_local)

        edges = Topology.generate_edges()
        cls._rounds = GreedyColoring.color_edges(edges, Topology.NTILE)
        cls._schedule = GreedyColoring.build_schedule(cls._rounds, Topology.NTILE)

        cls._update_domain_jit = jax.jit(
            jax.vmap(cls._update_domain_single, axis_name="tile")
        )
        
        cls._boundary_communication_jit = jax.jit(
            jax.vmap(cls._boundary_communication_single, axis_name="tile")
        )

    # -------------------------------------------------
    # 共享逻辑 
    # -------------------------------------------------
    @classmethod
    def _apply_transform(cls, data, recv_dir, tid):
        """统一的数据旋转/翻转应用层"""
        is_ew = recv_dir in (Topology.E, Topology.W)
        n = cls.ny_local if is_ew else cls.nx_local
        raw = data[:, :n]

        t = raw.T[::-1, :] if tid == Topology.FLIP_I_TRANS else raw

        expected_shape0 = cls.ny_local if is_ew else cls.halo
        if t.shape[0] != expected_shape0:
            t = t.T
            
        return t

    # -------------------------------------------------
    # Domain Update
    # -------------------------------------------------
    @classmethod
    def _pack_edge(cls, u, d):
        h, n = cls.halo, cls._n_pad

        if d == Topology.E:   raw = u[h:-h, -2*h:-h].T
        elif d == Topology.W: raw = u[h:-h, h:2*h].T
        elif d == Topology.N: raw = u[-2*h:-h, h:-h]
        else:                 raw = u[h:2*h, h:-h]

        # 【优化】使用 jnp.pad 替代 concatenate，XLA 编译更友好
        cur = raw.shape[1]
        if cur < n:
            raw = jnp.pad(raw, ((0, 0), (0, n - cur)))
        return raw

    @classmethod
    def _update_domain_round(cls, u, round_idx, perm):
        tile_id = lax.axis_index("tile")
        h, n = cls.halo, cls._n_pad
        sched = cls._schedule[round_idx]

        send = jnp.zeros((h, n), dtype=u.dtype)

        for t in range(Topology.NTILE):
            entry = sched[t]
            if entry is None: continue
            d, _, _ = entry
            
            send = lax.cond(
                tile_id == t,
                lambda _: cls._pack_edge(u, d),
                lambda _: send,
                None
            )

        recv = lax.ppermute(send, "tile", perm)

        for t in range(Topology.NTILE):
            entry = sched[t]
            if entry is None: continue
            d, _, tid = entry
            arr = cls._apply_transform(recv, d, tid)

            def write(_):
                if d == Topology.E:   return u.at[h:-h, -h:].set(arr)
                elif d == Topology.W: return u.at[h:-h, :h].set(arr)
                elif d == Topology.N: return u.at[-h:, h:-h].set(arr)
                else:                 return u.at[:h, h:-h].set(arr)

            u = lax.cond(tile_id == t, write, lambda _: u, None)

        return u

    @classmethod
    def _update_domain_single(cls, u):
        for r, (_, perm) in enumerate(cls._rounds):
            u = cls._update_domain_round(u, r, perm)
        return u

    @classmethod
    def update_domain(cls, u):
        with cls.mesh:
            return cls._update_domain_jit(u)

    # -------------------------------------------------
    # C 网格边界棱线同步
    # -------------------------------------------------
    @classmethod
    def _pack_boundary_edge(cls, u, v, d):
        h, n = cls.halo, cls._n_pad

        if d == Topology.E:   line = v[h:-h, cls.IDX_E]
        elif d == Topology.W: line = v[h:-h, cls.IDX_W]
        elif d == Topology.N: line = u[cls.IDX_N, h:-h]
        else:                 line = u[cls.IDX_S, h:-h]

        raw = jnp.zeros((h, len(line)), dtype=u.dtype).at[0, :].set(line)

        # 【优化】使用 jnp.pad 替代 concatenate
        cur = raw.shape[1]
        if cur < n:
            raw = jnp.pad(raw, ((0, 0), (0, n - cur)))
        return raw

    @classmethod
    def _boundary_communication_round(cls, u, v, round_idx, perm):
        tile_id = lax.axis_index("tile")
        h, n = cls.halo, cls._n_pad
        sched = cls._schedule[round_idx]

        send = jnp.zeros((h, n), dtype=u.dtype)

        for t in range(Topology.NTILE):
            entry = sched[t]
            if entry is None: continue
            d, _, _ = entry
            
            send = lax.cond(
                tile_id == t,
                lambda _: cls._pack_boundary_edge(u, v, d),
                lambda _: send,
                None
            )

        recv = lax.ppermute(send, "tile", perm)

        for t in range(Topology.NTILE):
            entry = sched[t]
            if entry is None: continue
            d, _, tid = entry
            arr = cls._apply_transform(recv, d, tid)

            def write(_):
                if d == Topology.E:
                    val = ((v[h:-h, cls.IDX_E] + arr[:, 0]) / 2.0).astype(v.dtype)
                    return u, v.at[h:-h, cls.IDX_E].set(val)
                elif d == Topology.W:
                    val = ((v[h:-h, cls.IDX_W] + arr[:, 0]) / 2.0).astype(v.dtype)
                    return u, v.at[h:-h, cls.IDX_W].set(val)
                elif d == Topology.N:
                    val = ((u[cls.IDX_N, h:-h] + arr[0, :]) / 2.0).astype(u.dtype)
                    return u.at[cls.IDX_N, h:-h].set(val), v
                else:
                    val = ((u[cls.IDX_S, h:-h] + arr[0, :]) / 2.0).astype(u.dtype)
                    return u.at[cls.IDX_S, h:-h].set(val), v

            u, v = lax.cond(tile_id == t, write, lambda _: (u, v), None)

        return u, v

    @classmethod
    def _boundary_communication_single(cls, u, v):
        for r, (_, perm) in enumerate(cls._rounds):
            u, v = cls._boundary_communication_round(u, v, r, perm)
        return u, v

    @classmethod
    def boundary_communication(cls, u, v):
        with cls.mesh:
            return cls._boundary_communication_jit(u, v)
    
    # -------------------------------------------------
    # 打印调度
    # -------------------------------------------------
    @classmethod
    def print_schedule_info(cls):
        if cls._rounds is None:
            print("错误: Communication 尚未配置。请先调用 configure()。")
            return

        print(f"通信轮次数: {len(cls._rounds)}")
        dn = {Topology.E: 'E', Topology.W: 'W', Topology.N: 'N', Topology.S: 'S'}
        
        for i, (edges, _) in enumerate(cls._rounds):
            round_info = [(e[0] + 1, dn[e[1]], e[2] + 1, dn[e[3]]) for e in edges]
            print(f"  轮{i}: {round_info}")