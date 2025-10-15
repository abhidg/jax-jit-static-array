import numpy as np
import time
import sys

import jax.random as jrand
import jax.numpy as jnp
from jax_jit_static_array.matmul import matmul_jit, matmul_nonjit

key = jrand.key(0)
x = jnp.arange(1, 7, dtype=np.float32)
mul = (2.0, 3.0, 1.0, 2.0)

sys.stdout.write("nonjit ")
sys.stdout.flush()

t_nonjit_start = time.time()
for i in range(100_000):
    matmul_nonjit(x + jrand.normal(key, x.shape, dtype=np.float32), mul)
    if i % 2000 == 0:
        sys.stdout.write(".")
        sys.stdout.flush()
t_nonjit = time.time() - t_nonjit_start
print(f"\nnonjit {t_nonjit:.2f}s")

t_jit_start = time.time()
sys.stdout.write("   jit ")
sys.stdout.flush()
for i in range(100_000):
    matmul_jit(x + jrand.normal(key, x.shape, dtype=np.float32), mul)
    if i % 2000 == 0:
        sys.stdout.write(".")
        sys.stdout.flush()
t_jit = time.time() - t_jit_start
print(f"\n   jit {t_jit:.2f}s")

if t_jit > t_nonjit:
    print("JIT is usually faster, so this is being considered a failure")
    sys.exit(1)
