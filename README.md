# JAX JIT with a static array

Demo showing JAX JIT with a static array being passed to the C++ backend.

Starting from a simplified version of the `rms_norm` example shown in [JAX FFI docs](https://docs.jax.dev/en/latest/ffi.html). The linked code example from the docs is more complex with type-based dispatch. I have modified the `rms_norm` code to match the simpler snippets that only work on `float32`, and removed the dependency on absl-py as pytest is sufficient.

Added an example with matrix multiplication (`matmul`), where the first two values of a static argument (`mul`) are nrows, ncols of the first array `x`, and the rest is the row vector that will be left-multiplied onto `x`. This shows static arrays being passed to JAX, and `matmul_perf.py` shows the performance gains when JIT is applied.
