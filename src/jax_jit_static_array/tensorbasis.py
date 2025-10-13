import jax
import jax.numpy as jnp
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorBasis:
    width: int
    depth: int
    degree_begin: bytes


def make_basis(width: int, depth: int) -> TensorBasis:
    data = np.zeros(depth + 2, dtype=np.int32)
    for i in range(1, depth + 2):
        data[i] = 1 + width * data[i - 1]
    return TensorBasis(width, depth, data.tobytes())


def ft_altsquare(x, basis: TensorBasis):
    if x.dtype != jnp.float32:
        raise ValueError(
            "Only the float32 dtype is implemented by alt_square_free_tensor"
        )

    call = jax.ffi.ffi_call(
        "ft_altsquare",
        jax.ShapeDtypeStruct(x.dtype, x.dtype),
    )
    return call(x, basis)


ft_altsquare_jit = jax.jit(ft_altsquare, static_argnames=["basis"])
