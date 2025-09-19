# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp

import numpy.testing as npt
from jax_jit_static_array import rms_norm


def rms_norm_ref(x, eps=1e-5):
    eps = jnp.float32(eps).astype(x.dtype)
    scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x / scale


def test_basic():
    x = jnp.linspace(-0.5, 0.5, 15)
    npt.assert_allclose(rms_norm.rms_norm(x), rms_norm_ref(x), rtol=1e-5)


def test_batching():
    x = jnp.linspace(-0.5, 0.5, 15).reshape((3, 5))
    npt.assert_allclose(
        jax.vmap(rms_norm.rms_norm)(x), jax.vmap(rms_norm_ref)(x), rtol=1e-5
    )
