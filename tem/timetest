import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P

# 导入你重构好的通信类和分发类
from communication import Communication
from gpu_mesh import Global2Local 

# =====================================
# 主程序：K轮 update_domain 性能测试
# =====================================
def main():
    # 1. 基础配置
    ntile    = 6
    halo     = 3
    nx       = 1536
    ny       = nx
    pdev, px, py   = 6, 1, 1
    nx_local = nx // px
    ny_local = ny // py

    ndev = len(jax.devices())
    assert ndev == pdev, f"配置需要 {pdev} 张卡，当前检测到 {ndev} 张"

    # 2. 网格与分片初始化
    devices     = np.array(jax.devices()).reshape(pdev, 1, 1)
    mesh        = Mesh(devices, ('tile', 'x', 'y'))
    sharding_2d = NamedSharding(mesh, P('tile', 'x', 'y'))

    # 3. 配置通信与本地化拓扑
    Global2Local.configure(sharding_2d, halo, nx_local, ny_local)
    Communication.configure(halo, nx_local, ny_local, mesh)
    
    print("-" * 40)
    Communication.print_schedule_info()
    print("-" * 40)

    # 4. 初始化全局数据
    tile = jnp.arange(ntile)[:, None, None]
    i_idx = jnp.arange(ny)[None, :, None]
    j_idx = jnp.arange(nx)[None, None, :]
    u_global = ((tile + 1) * 10000 + i_idx * 100 + j_idx) * 1.0

    # 5. 分发数据到本地卡
    u_dist = Global2Local.distribute(u_global)
    print(f"数据初始化完成: u_global{u_global.shape} -> u_dist{u_dist.shape}")

    # =====================================
    # 性能测试 (Benchmark)
    # =====================================
    k_rounds = 2000  # 定义要测试的轮数

    print("\n[1/2] 正在预热 JIT 编译器...")
    warmup_start = time.perf_counter()
    u_run = Communication.update_domain(u_dist)
    u_run.block_until_ready()  # 必须阻塞以确保编译和第一次运行彻底完成
    warmup_end = time.perf_counter()
    print(f"预热耗时 (包含 JIT 编译): {warmup_end - warmup_start:.4f} 秒")

    print(f"\n[2/2] 开始测试 {k_rounds} 轮 update_domain 执行时间...")
    
    # 为了保证公平，用一个新的变量开始计时循环
    u_test = u_run 
    
    # 确保 GPU 处于空闲状态后再开始计时
    jax.effects_barrier() 
    start_time = time.perf_counter()

    for _ in range(k_rounds):
        u_test = Communication.update_domain(u_test)
        
    # 等待所有提交的内核在 GPU 上执行完毕
    u_test.block_until_ready()
    end_time = time.perf_counter()

    # 6. 统计与打印结果
    total_time = end_time - start_time
    avg_time_ms = (total_time / k_rounds) * 1000

    print("-" * 40)
    print("性能测试结果:")
    print(f"总计算轮数:     {k_rounds}")
    print(f"网格分辨率:     {ntile} 个 tile, 每个 {nx}x{ny} (加 Halo)")
    print(f"总消耗时间:     {total_time:.4f} 秒")
    print(f"单轮平均耗时:   {avg_time_ms:.4f} 毫秒/轮")
    print("-" * 40)


if __name__ == "__main__":
    main()