from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from jax_jit_static_array import _tensorbasis

for name, target in _tensorbasis.registrations().items():
  jax.ffi.register_ffi_target(name, target)

T = jnp.array([2, 1, 3, 0.5, -1, 2, 0], dtype=np.float32)

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
    degree_begin = jnp.frombuffer(basis.degree_begin, dtype=jnp.int32)
    if x.dtype != jnp.float32:
        raise ValueError(
            "Only the float32 dtype is implemented by alt_square_free_tensor"
        )

    call = jax.ffi.ffi_call(
        "alt_square_free_tensor",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    return call(x, degree_begin, size=int(x.shape[0]))


ft_altsquare_jit = jax.jit(ft_altsquare, static_argnames=["basis"])

if __name__ == "__main__":
    print("in tensorbasis module __main__")
    print(ft_altsquare(T, make_basis(2, 2)))
    print(ft_altsquare_jit(T, make_basis(2, 2)))
