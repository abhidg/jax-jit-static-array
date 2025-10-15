import numpy as np
import time
import sys

import jax.random as jrand
import jax.numpy as jnp
from jax_jit_static_array.tensorbasis import ft_altsquare, ft_altsquare_jit, make_basis

key = jrand.key(0)
x = jnp.arange(1, 8, dtype=np.float32)
basis = make_basis(2, 2)  # width, depth

sys.stdout.write("nonjit ")
sys.stdout.flush()

t_nonjit_start = time.time()
for i in range(100_000):
    ft_altsquare(x + jrand.normal(key, x.shape, dtype=jnp.float32), basis)
    if i % 2000 == 0:
        sys.stdout.write(".")
        sys.stdout.flush()
t_nonjit = time.time() - t_nonjit_start
print(f"\nnonjit {t_nonjit:.2f}s")

t_jit_start = time.time()
sys.stdout.write("   jit ")
sys.stdout.flush()
for i in range(100_000):
    ft_altsquare_jit(x + jrand.normal(key, x.shape, dtype=jnp.float32), basis)
    if i % 2000 == 0:
        sys.stdout.write(".")
        sys.stdout.flush()
t_jit = time.time() - t_jit_start
print(f"\n   jit {t_jit:.2f}s")

if t_jit > t_nonjit:
    print("JIT is usually faster, so this is being considered a failure")
    sys.exit(1)
