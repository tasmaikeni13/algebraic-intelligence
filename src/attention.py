"""Octic attention using rational operations and reciprocal square roots.

Last-axis normalization, FP32 accumulation for half inputs, input/output dtype
preservation. Finite squared reductions and representable octic powers are
required; AVN bounds coordinates by sqrt(length). eps and sink_omega are static.
The entrywise Jacobian bound of 2 applies to normalized scores, not raw scores
or the spectral norm. No uniform quantization advantage is implied.
"""
from functools import partial

import jax
import jax.numpy as jnp

from src.primitives import _accumulate, _avn_forward, _avn_backward


def _kernel(z):
    r = jax.lax.rsqrt(1.0 + z * z)
    u = z * r
    # Conjugate identity avoids cancellation when z is negative.
    denominator = jnp.where(z < 0, 1.0 - u, 1.0)
    rho = jnp.where(z < 0, r / denominator, z + (1.0 + z * z) * r)
    k2 = rho * rho
    k4 = k2 * k2
    return k4 * k4, r


def _octic_forward(x):
    k, r = _kernel(_accumulate(x))
    return k.astype(x.dtype), (k, r)


@jax.custom_vjp
def _octic(x):
    return _octic_forward(x)[0]


def _octic_backward(cache, g):
    k, r = cache
    return ((8.0 * r * k * g.astype(k.dtype)).astype(g.dtype),)


_octic.defvjp(_octic_forward, _octic_backward)


@jax.jit
def octic_kernel(x):
    """rho(x)^8 via exactly three squarings after evaluating rho."""
    return _octic(jnp.asarray(x))


def _normalized_forward(z, sink_omega):
    k, r = _kernel(z)
    p = k / (jnp.sum(k, axis=-1, keepdims=True) + sink_omega)
    return p, r


def _attention_forward(x, sink_omega, eps):
    if not 0 <= sink_omega < float("inf"):
        raise ValueError("sink_omega must be finite and nonnegative")
    if not 0 < eps < float("inf"):
        raise ValueError("eps must be finite and positive")
    z = _accumulate(x)
    normalized, (_, tau) = _avn_forward(z, eps)
    p, r = _normalized_forward(normalized, sink_omega)
    return p.astype(x.dtype), (normalized, tau, p, r)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _attention(x, sink_omega, eps):
    return _attention_forward(x, sink_omega, eps)[0]


def _attention_backward(sink_omega, eps, cache, g):
    del sink_omega
    normalized, tau, p, r = cache
    upstream = g.astype(p.dtype)
    dot = jnp.sum(p * upstream, axis=-1, keepdims=True)
    normalized_g = 8.0 * r * p * (upstream - dot)
    dx = _avn_backward(eps, (normalized, tau), normalized_g)[0]
    return (dx.astype(g.dtype),)


_attention.defvjp(_attention_forward, _attention_backward)


@partial(jax.jit, static_argnames=("sink_omega", "eps"))
def algebraic_softmax(scores, sink_omega=0.5, eps=1e-5):
    """AVN-bounded octic attention; omitted mass belongs to a rational sink."""
    return _attention(jnp.asarray(scores), sink_omega, eps)
