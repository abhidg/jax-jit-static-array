# JAX JIT with a static array

Demo showing JAX JIT with a static array being passed to the C++ backend.

Starting from a simplified version of the `rms_norm` example shown in [JAX FFI docs](https://docs.jax.dev/en/latest/ffi.html). The linked code example from the docs is more complex with type-based dispatch. I have modified the `rms_norm` code to match the simpler snippets that only work on `float32`, and removed the dependency on absl-py as pytest is sufficient.

Other modules:

- **matmul**: Example with matrix multiplication (`matmul`), where the first
  two values of a static argument (`mul`) are nrows, ncols of the first array
  `x`, and the rest is the row vector that will be left-multiplied onto `x`.
  This shows static arrays being passed to JAX, and `matmul_perf.py` shows the
  performance gains when JIT is applied.
- **tensorbasis**: Shows an example passing a struct to JAX with the
  TensorBasis class used in RoughPy. The module computes a function that
  squares the elements of a free tensor at alternate depths. So a free tensor
  at width=2, depth=2 would have seven values e.g. [1, 0.5, 3, -1, 4, 9, 0],
  which this module would square and return [1, 0.25, 9, -1, 4, 9, 0], squaring
  elements at depth 1 only. This example shows that TensorBasis can be JIT
  compiled and used for computation at the XLA FFI layer. To ensure caching,
  the `degree_begin` array is passed as bytes.
