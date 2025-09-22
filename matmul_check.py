from jax_jit_static_array.matmul import matmul_jit, matmul_nonjit
import numpy as np
import jax.random as jrand
import jax.numpy as jnp
import time
import sys
key = jrand.key(0)
x = jnp.arange(1, 7, dtype=np.float32)
mul = (2.0, 3.0, 1.0, 2.0)

print("Non-JIT version:")
t_nonjit = time.time()
for i in range(100_000):
    matmul_nonjit(x + jrand.normal(key, x.shape), mul)
    if i % 1000 == 0:
        sys.stdout.write(".")
        sys.stdout.flush()
print("\nTime elapsed (non-JIT): ", time.time() - t_nonjit)

t_jit = time.time()
print("JIT version:")
for i in range(100_000):
    matmul_jit(x + jrand.normal(key, x.shape), mul)
    if i % 1000 == 0:
        sys.stdout.write(".")
        sys.stdout.flush()
print("\nTime elapsed (JIT): ", time.time() - t_jit)
