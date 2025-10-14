import jax.numpy as jnp
import numpy.testing as npt
from jax_jit_static_array.tensorbasis import ft_altsquare, ft_altsquare_jit, make_basis

T = jnp.array([2, 0.5, 3, 0.5, -1, 2, 0], dtype=jnp.float32)
basis = make_basis(2, 2)  # width, depth

# alternate depths are squared
T_alt_squared = jnp.array([2, 0.25, 9, 0.5, -1, 2, 0], dtype=jnp.float32)


def test_jit():
    npt.assert_array_equal(ft_altsquare_jit(T, basis), T_alt_squared)

def test_nonjit():
    npt.assert_array_equal(ft_altsquare(T, basis), T_alt_squared)
