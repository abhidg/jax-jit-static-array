import numpy as np
import numpy.testing as npt

import jax.random as jrand
import jax.numpy as jnp
from jax_jit_static_array.matmul import matmul_jit, matmul_nonjit

key = jrand.key(0)
x = jnp.arange(1, 7, dtype=np.float32)
mul = (2.0, 3.0, 1.0, 2.0)
expected = np.array([5, 11, 17])

def test_matmul_nonjit():
    y = matmul_nonjit(x, mul)
    npt.assert_array_equal(y[0], expected)

def test_matmul_jit():
    y = matmul_jit(x, mul)
    npt.assert_array_equal(y[0], expected)
