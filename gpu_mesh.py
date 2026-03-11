import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

# =====================================
# Global2Local Class
# =====================================
class Global2Local:
    """
    Distributes global arrays to individual devices and adds zero-padded 
    halo boundaries to each tile.

    Usage uses **class attributes + class methods**, eliminating the need 
    for instantiation:

    1. First, call `Global2Local.configure(...)` to complete the global configuration.
    2. Then, call `Global2Local.distribute(global_data)` to execute the distribution.
    """

    # Class attributes (Global configuration)
    sharding: NamedSharding | None = None
    halo: int = 0
    nx_local: int = 0
    ny_local: int = 0
    
    # JIT-compiled vmap function for adding halos
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
        Configures the global parameters. Acts as a replacement for __init__.
        
        Args:
            sharding: JAX NamedSharding specification for device placement.
            halo: Width of the halo (ghost cell) boundary.
            nx_local: Number of local grid points in the x-direction per tile (excluding halo).
            ny_local: Number of local grid points in the y-direction per tile (excluding halo).
        """
        cls.sharding = sharding
        cls.halo = halo
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        
        # Vectorize the single-tile halo addition across the leading axis (ntile)
        # JIT compilation ensures the padding operation is optimized for the accelerator
        cls._add_halo_vmap = jax.jit(jax.vmap(cls._add_halo_single))

    @classmethod
    def _add_halo_single(cls, data_in: jnp.ndarray) -> jnp.ndarray:
        """
        Pads a single tile's data with a zero-filled halo region.
        """
        h = cls.halo
        # Initialize a zero array with dimensions expanded by the halo width on both sides
        u = jnp.zeros(
            (cls.nx_local + 2 * h, cls.ny_local + 2 * h),
            dtype=data_in.dtype,
        )
        # Inject the original data into the center, leaving the halo region as zeros
        return u.at[h:-h, h:-h].set(data_in)

    @classmethod
    def distribute(cls, global_data: jnp.ndarray) -> jnp.ndarray:
        """
        Places the global data onto the specified devices and applies halo padding.

        Args:
            global_data: The global array with shape (ntile, nx, ny).
            
        Returns:
            The distributed and padded array with shape (ntile, nx_local + 2*h, ny_local + 2*h).
        """
        # Distribute the data across the specified device mesh
        data_dist = jax.device_put(global_data, cls.sharding)
        
        # Apply the vectorized halo padding across all tiles
        return cls._add_halo_vmap(data_dist)