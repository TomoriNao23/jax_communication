from communication import HaloExchange
from gpu_mesh import Global2Local
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P

# =====================================
# 主程序（单卡版：用1张卡模拟6个tile）
# =====================================
def main():
    ntile    = 6
    halo     = 3
    nx       = 192
    ny       = 192
    pdev, px, py   = 1, 1, 1
    nx_local = nx // px
    ny_local = ny // py

    ndev      = len(jax.devices())
    assert ndev == pdev

    # tile轴直接用实际设备数，1卡时tile=1，6卡时tile=6
    devices     = np.array(jax.devices()).reshape(pdev, 1, 1)
    mesh        = Mesh(devices, ('tile', 'x', 'y'))
    sharding_2d = NamedSharding(mesh, P('tile', 'x', 'y'))

    # 使用类配置方式初始化 Global2Local 和 HaloExchange
    Global2Local.configure(sharding_2d, halo, nx_local, ny_local)
    HaloExchange.configure(halo, nx_local, ny_local, mesh)

    print(f"通信轮次数: {len(HaloExchange._rounds)}")
    for i, (edges, perm) in enumerate(HaloExchange._rounds):
        E2, W2, N2, S2 = HaloExchange.E, HaloExchange.W, HaloExchange.N, HaloExchange.S
        dn = {E2:'E', W2:'W', N2:'N', S2:'S'}
        print(f"  轮{i}: {[(e[0],dn[e[1]],e[2],dn[e[3]]) for e in edges]}")

    # 每个 tile 填充常数 (i+1)
    u_global = jnp.stack([jnp.full((nx, ny), float(i + 1)) for i in range(ntile)])

    # 直接通过类方法完成分发
    u_dist  = Global2Local.distribute(u_global)
    u_final = HaloExchange.exchange(u_dist)
    u_final.block_until_ready()

    h = halo
    print("\n=== Halo 验证（边 ID 对应的直接接触，期望整数值）===")
    checks = [
        # (描述, tile_idx, halo切片, 期望值)
        ("Tile 1 W halo ← Tile 0 E (期望 1.0)", 1, (slice(h,-h), slice(None,h)),  1.0),
        ("Tile 0 E halo ← Tile 1 W (期望 2.0)", 0, (slice(h,-h), slice(-h,None)), 2.0),
        ("Tile 4 S halo ← Tile 0 N (期望 1.0)", 4, (slice(None,h), slice(h,-h)),  1.0),
        ("Tile 0 N halo ← Tile 4 S (期望 5.0)", 0, (slice(-h,None), slice(h,-h)), 5.0),
        ("Tile 5 N halo ← Tile 0 S (期望 1.0)", 5, (slice(-h,None), slice(h,-h)), 1.0),
        ("Tile 0 S halo ← Tile 5 N (期望 6.0)", 0, (slice(None,h), slice(h,-h)),  6.0),
        ("Tile 2 W halo ← Tile 1 E (期望 2.0)", 2, (slice(h,-h), slice(None,h)),  2.0),
        ("Tile 3 W halo ← Tile 2 E (期望 3.0)", 3, (slice(h,-h), slice(None,h)),  3.0),
    ]

    all_pass = True
    for desc, tile, slices, expected in checks:
        region   = u_final[tile][slices]
        mean_val = float(jnp.mean(region))
        ok       = jnp.allclose(region, expected)
        status   = "✓" if ok else "✗"
        print(f"  {status} {desc}: mean={mean_val:.4f}")
        if not ok:
            all_pass = False

    print(f"\n{'所有验证通过 ✓' if all_pass else '存在失败项 ✗'}")

if __name__ == "__main__":
    main()