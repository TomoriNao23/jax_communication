import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
# =====================================
# Global2Local 类
# =====================================
class Global2Local:
    """
    将全局数组分发到各设备，并为每个 tile 添加零 halo 边界。

    使用方式改为**类属性 + 类方法**，不再需要实例化：

    1. 先调用 `Global2Local.configure(...)` 完成全局配置
    2. 再调用 `Global2Local.distribute(global_data)` 完成分发

    Args (for configure):
        sharding  : JAX NamedSharding
        halo      : halo 宽度
        nx_local  : 单 tile 本地 x 方向网格点数（不含 halo）
        ny_local  : 单 tile 本地 y 方向网格点数（不含 halo）
    """

    # 类属性（全局配置）
    sharding: NamedSharding | None = None
    halo: int = 0
    nx_local: int = 0
    ny_local: int = 0
    _add_halo_vmap = None

    @classmethod
    def configure(
        cls,
        sharding: NamedSharding,
        halo: int,
        nx_local: int,
        ny_local: int,
    ) -> None:
        """
        配置全局参数，相当于原来的 __init__。
        """
        cls.sharding = sharding
        cls.halo = halo
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls._add_halo_vmap = jax.jit(jax.vmap(cls._add_halo_single))

    @classmethod
    def _add_halo_single(cls, data_in: jnp.ndarray) -> jnp.ndarray:
        h = cls.halo
        u = jnp.zeros(
            (cls.nx_local + 2 * h, cls.ny_local + 2 * h),
            dtype=data_in.dtype,
        )
        return u.at[h:-h, h:-h].set(data_in)

    @classmethod
    def distribute(cls, global_data: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            global_data: (ntile, nx, ny)
        Returns:
            (ntile, nx_local+2h, ny_local+2h)
        """
        data_dist = jax.device_put(global_data, cls.sharding)
        return cls._add_halo_vmap(data_dist)