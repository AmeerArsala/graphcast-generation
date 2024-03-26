import jax
import jaxlib
import jax.numpy as jnp


ATMOSPHERIC_PRESSURE_LEVELS = jnp.array(
    [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=jnp.int32
)
