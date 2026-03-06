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

    # -------------------------------------------------
    # transform 判断
    # -------------------------------------------------

    @classmethod
    def _infer_transform(cls, dA, dB):

        table = {
            # 不旋转 (共线拼接面)
            (cls.E, cls.W): cls.ID,
            (cls.W, cls.E): cls.ID,
            (cls.N, cls.S): cls.ID,
            (cls.S, cls.N): cls.ID,

            # 转置 + 反转边界长度维度 (跨接拼接面)
            # 在当前六面体展开拓扑中，所有跨接面映射互为逆运算，均使用此变换
            (cls.E, cls.S): cls.FLIP_I_TRANS,
            (cls.W, cls.N): cls.FLIP_I_TRANS,
            (cls.S, cls.E): cls.FLIP_I_TRANS,
            (cls.N, cls.W): cls.FLIP_I_TRANS,
        }

        if (dA, dB) not in table:
            raise ValueError(f"invalid orientation {dA} -> {dB}")

        return table[(dA, dB)]

    # -------------------------------------------------
    # 生成拓扑
    # -------------------------------------------------

    @classmethod
    def _generate_edges(cls):

        edges = []

        for tile in range(cls.ntile):

            if tile % 2 == 0:
                # real odd

                rules = {
                    cls.S: ((tile + 4) % 6, cls.E),
                    cls.E: ((tile + 2) % 6, cls.S),
                    cls.N: ((tile + 1) % 6, cls.S),
                    cls.W: ((tile + 5) % 6, cls.E),
                }

            else:
                # real even

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

    # -------------------------------------------------
    # 图着色
    # -------------------------------------------------

    @classmethod
    def _color_edges(cls):

        n = len(cls.EDGES)

        conflicts = [[False]*n for _ in range(n)]

        for i,(a,_,b,_,_) in enumerate(cls.EDGES):
            for j,(c,_,d,_,_) in enumerate(cls.EDGES):

                if i!=j and {a,b}&{c,d}:
                    conflicts[i][j]=True

        colors=[-1]*n

        for i in range(n):

            used={colors[j] for j in range(n)
                  if conflicts[i][j] and colors[j]>=0}

            c=0
            while c in used:
                c+=1

            colors[i]=c

        rounds=[]

        for c in range(max(colors)+1):

            group=[cls.EDGES[i] for i in range(n) if colors[i]==c]

            swapped=set()

            for a,_,b,_,_ in group:
                swapped.add(a)
                swapped.add(b)

            perm=[]

            for a,_,b,_,_ in group:
                perm.append((a,b))
                perm.append((b,a))

            for t in range(cls.ntile):
                if t not in swapped:
                    perm.append((t,t))

            rounds.append((group,perm))

        return rounds

    # -------------------------------------------------

    @classmethod
    def _build_schedule(cls):

        schedule=[]

        for group,_ in cls._rounds:

            round_sched=[None]*cls.ntile

            for tA,dA,tB,dB,tid in group:

                round_sched[tA]=(dA,tB,tid)
                round_sched[tB]=(dB,tA,tid)

            schedule.append(round_sched)

        return schedule

    # -------------------------------------------------

    @classmethod
    def _pack_edge(cls,u,d):

        h=cls.halo
        n=cls._n_pad

        if d==cls.E:
            raw=u[h:-h,-2*h:-h].T
        elif d==cls.W:
            raw=u[h:-h,h:2*h].T
        elif d==cls.N:
            raw=u[-2*h:-h,h:-h]
        else:
            raw=u[h:2*h,h:-h]

        cur=raw.shape[1]

        if cur<n:
            raw=jnp.concatenate(
                [raw,jnp.zeros((h,n-cur),dtype=raw.dtype)],axis=1
            )

        return raw

    # -------------------------------------------------

    @classmethod
    def _apply_transform(cls, data, recv_dir, tid):

        if recv_dir in (cls.E, cls.W):
            n = cls.ny_local
        else:
            n = cls.nx_local

        raw = data[:, :n]

        # --- cubed sphere orientation group ---

        if tid == cls.ID:
            t = raw

        elif tid == cls.FLIP_I_TRANS:
            # transpose + reverse boundary length dimension
            # 保留 halo 深度维度不被翻转
            t = raw.T[::-1, :]

        else:
            raise ValueError("invalid transform")

        # --- orient halo correctly ---

        if recv_dir in (cls.E, cls.W):

            # need (ny_local, h)
            if t.shape[0] != cls.ny_local:
                t = t.T

        else:

            # need (h, nx_local)
            if t.shape[0] != cls.halo:
                t = t.T

        return t

    # -------------------------------------------------
    @classmethod
    def _one_round(cls,u,round_idx,perm):

        tile_id=lax.axis_index("tile")

        h=cls.halo
        n=cls._n_pad

        sched=cls._schedule[round_idx]

        send=jnp.zeros((h,n),dtype=u.dtype)

        for t in range(cls.ntile):

            entry=sched[t]

            if entry is None:
                continue

            d,_,_=entry

            send=lax.cond(
                tile_id==t,
                lambda _:cls._pack_edge(u,d),
                lambda _:send,
                None
            )

        recv=lax.ppermute(send,"tile",perm)

        for t in range(cls.ntile):

            entry=sched[t]

            if entry is None:
                continue

            d,_,tid=entry

            arr=cls._apply_transform(recv,d,tid)

            def write(_):

                if d==cls.E:
                    return u.at[h:-h,-h:].set(arr)
                elif d==cls.W:
                    return u.at[h:-h,:h].set(arr)
                elif d==cls.N:
                    return u.at[-h:,h:-h].set(arr)
                else:
                    return u.at[:h,h:-h].set(arr)

            u=lax.cond(tile_id==t,write,lambda _:u,None)

        return u

    # -------------------------------------------------

    @classmethod
    def _exchange_single(cls,u):

        for r,(_,perm) in enumerate(cls._rounds):
            u=cls._one_round(u,r,perm)

        return u

    # -------------------------------------------------

    @classmethod
    def exchange(cls,u):

        with cls.mesh:
            return cls._exchange_jit(u)
    
    # -------------------------------------------------
    # 打印通信计划 (用于测试和验证)
    # -------------------------------------------------

    @classmethod
    def print_schedule_info(cls):
        """打印每一轮通信的具体面片连接情况"""
        if cls._rounds is None:
            print("错误: HaloExchange 尚未配置。请先调用 configure()。")
            return

        print(f"通信轮次数: {len(cls._rounds)}")
        
        # 方向数字到字符的映射
        dn = {cls.E: 'E', cls.W: 'W', cls.N: 'N', cls.S: 'S'}
        
        for i, (edges, _) in enumerate(cls._rounds):
            # e 的结构是 (tileA, dirA, tileB, dirB, tid)
            # 转换为 (tileA+1, 方向A, tileB+1, 方向B)
            round_info = [
                (e[0] + 1, dn[e[1]], e[2] + 1, dn[e[3]]) 
                for e in edges
            ]
            print(f"  轮{i}: {round_info}")