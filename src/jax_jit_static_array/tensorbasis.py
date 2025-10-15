from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from jax_jit_static_array import _tensorbasis

# https://docs.jax.dev/en/latest/default_dtypes.html#the-x64-flag-enabling-64-bit-values
# NOTE: This is a global flag and WILL affect code subsequent to tensorbasis module import
#       Currently JAX does not support contextual configuration, see above link
jax.config.update('jax_enable_x64', True)

for name, target in _tensorbasis.registrations().items():
  jax.ffi.register_ffi_target(name, target)

T = jnp.array([2, 1, 3, 0.5, -1, 2, 0], dtype=np.float32)

@dataclass(frozen=True)
class TensorBasis:
    width: int
    depth: int
    degree_begin: np.ndarray

    def __hash__(self):
        # This works for the tensor basis as the data is entirely determined
        # by the tuple (width, depth)
        return hash((self.width, self.depth))


def make_basis(width: int, depth: int) -> TensorBasis:
    data = np.zeros(depth + 2, dtype=np.int64)
    for i in range(1, depth + 2):
        data[i] = 1 + width * data[i - 1]
    return TensorBasis(width, depth, data)

def ft_altsquare(x, basis: TensorBasis):
    if x.dtype != jnp.float32:
        raise ValueError(
            "Only the float32 dtype is implemented by alt_square_free_tensor"
        )

    call = jax.ffi.ffi_call(
        "alt_square_free_tensor",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    return call(x, basis.degree_begin, size=int(x.shape[0]))


ft_altsquare_jit = jax.jit(ft_altsquare, static_argnames=["basis"])

if __name__ == "__main__":
    print("in tensorbasis module __main__")
    print(ft_altsquare(T, make_basis(2, 2)))
    print(ft_altsquare_jit(T, make_basis(2, 2)))
