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
    nx       = 96
    ny       = nx
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
    HaloExchange.print_schedule_info()

    # 每个 tile 填充常数 (i+1)
    u_global = jnp.zeros((ntile,nx,ny),dtype=jnp.float64)
    tile = jnp.arange(ntile)[:,None,None]
    i_idx = jnp.arange(ny)[None,:, None]
    j_idx = jnp.arange(nx)[None,None,:]
    u_global = (tile + 1) * 10000 + i_idx * 100 + j_idx
    # 直接通过类方法完成分发
    u_dist  = Global2Local.distribute(u_global)
    print(u_global.shape,u_dist.shape)

    sum_per_tile = jnp.sum(u_dist, axis=(1, 2))
    sum_per_tile_host = jax.device_get(sum_per_tile)
    print(sum_per_tile_host)

    # vars = u_dist[:, 2, 3]
    # print(jax.device_get(vars))

    u_final = HaloExchange.exchange(u_dist)
    u_final.block_until_ready()

    sum_per_tile = jnp.sum(u_final, axis=(1, 2))
    sum_per_tile_host = jax.device_get(sum_per_tile)
    print(sum_per_tile_host)

    

    
if __name__ == "__main__":
    main()