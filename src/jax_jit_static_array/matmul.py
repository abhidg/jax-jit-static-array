import jax
import jax.numpy as jnp
from functools import partial

from jax_jit_static_array import _matmul

for name, target in _matmul.registrations().items():
  jax.ffi.register_ffi_target(name, target)

@partial(jax.jit, static_argnames=["mul"])
def matmul_jit(x, mul: tuple):
  if x.dtype != jnp.float32:
    raise ValueError("Only the float32 dtype is implemented by rms_norm")

  nrows, ncols = mul[:2]
  nrows = int(nrows)
  ncols = int(ncols)
  if len(mul) != nrows + 2:
    raise ValueError("Invalid mul supplied, number of elements must be the first index + 2")
  call = jax.ffi.ffi_call(
    "matmul",
    jax.ShapeDtypeStruct((1, ncols), x.dtype),
  )
  return call(x, jnp.array(mul))

def matmul_nonjit(x, mul: tuple):
  if x.dtype != jnp.float32:
    raise ValueError("Only the float32 dtype is implemented by rms_norm")

  nrows, ncols = mul[:2]
  nrows = int(nrows)
  ncols = int(ncols)
  if len(mul) != nrows + 2:
    raise ValueError("Invalid mul supplied, number of elements must be the first index + 2")
  call = jax.ffi.ffi_call(
    "matmul",
    jax.ShapeDtypeStruct((1, ncols), x.dtype),
  )
  return call(x, jnp.array(mul))
