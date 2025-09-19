# JAX JIT with a static array

Demo showing JAX JIT with a static array being passed to the C++ backend.

Starting from a simplified version of the `rms_norm` example shown in [JAX FFI docs](https://docs.jax.dev/en/latest/ffi.html). The linked code example from the docs is more complex with type-based dispatch. I have modified the `rms_norm` code to match the simpler snippets that only work on `float32`, and removed the dependency on absl-py as pytest is sufficient.
