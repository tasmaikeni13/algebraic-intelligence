"""Phase 7: Full Architecture Assembly — Algebraic Transformer (AlgebraicTransformerLM).

Integrates verified Phase 1-6 algebraic primitives into a unified causal language model:
- Token embedding with parameter-free AVN
- Stack of 6 Algebraic Blocks:
  - Parameter-free AVN pre-normalization
  - Octic A-Softmax (rho^8) + AGO Cayley rotations + Pallas AFA / causal algebraic attention
  - Additive residual connections
  - Parameter-free AVN pre-normalization
  - ALU-GLU feed-forward network with Horner cubic backward
  - Additive residual connections
- Final parameter-free AVN
- Linear un-embedding head
- Octic Algebraic Cross-Entropy (OACE / L_1/8) proper scoring loss functional via 3 hardware rsqrt
- Optimizable via natively algebraic AdamW + ARDS rational decay schedule

Strictly zero transcendental functions (zero exp, zero log, zero sin, zero cos).
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp

from src.primitives import alu, avn, _accumulate, _avn_forward, _avn_backward
from src.attention import (
    algebraic_softmax,
    apply_ago_rotations,
    build_cayley_rotary_matrix,
    CayleyRotary,
    _octic,
    _kernel,
    _extract_cs,
    _align_param,
    _rotate_tensor,
)
from src.kernels.pallas_afa import _vmu_octic_kernel
from src.loss import oace_loss
from src.baseline import StandardTransformerLM, BaselineConfig


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for 15M pilot scale language models."""
    vocab_size: int = 50257
    d_model: int = 288
    num_layers: int = 6
    num_heads: int = 6
    d_ff: int = 768
    max_seq_len: int = 512
    eps: float = 1e-5
    eps_vocab: float = 100.0
    sink_omega: float = 0.5
    gamma: float = 2.0
    tie_embeddings: bool = True
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32


def count_parameters(params: Any) -> int:
    """Counts the total number of scalar parameters in a PyTree."""
    leaves = jax.tree_util.tree_leaves(params)
    return sum(x.size for x in leaves)


def _fast_alu_fwd(x):
    s_sq = x * x
    one = jnp.array(1.0, dtype=x.dtype)
    rad = one + s_sq
    r = jax.lax.rsqrt(rad)
    u = x * r
    denom = jnp.where(x < 0, one - u, one)
    neg = (0.5 * u * r) / denom
    pos = 0.5 * x * (one + u)
    y = jnp.where(x < 0, neg, pos)
    return y, u


def _fast_alu_bwd(u, g):
    half = jnp.array(0.5, dtype=g.dtype)
    one = jnp.array(1.0, dtype=g.dtype)
    deriv = half + u * (one - half * (u * u))
    return (g * deriv,)


@jax.custom_vjp
def _fast_alu(x):
    return _fast_alu_fwd(x)[0]


_fast_alu.defvjp(_fast_alu_fwd, _fast_alu_bwd)


def _alu_glu(x: jax.Array, w_g: jax.Array, w_u: jax.Array, w_d: jax.Array) -> jax.Array:
    """Evaluates ALU-GLU: W_d [ (W_g x) * ALU(W_u x) ] in native hardware precision."""
    gate = jnp.matmul(x, w_g)
    up = jnp.matmul(x, w_u)
    activated = _fast_alu(up)
    return jnp.matmul(gate * activated, w_d)


def _causal_algebraic_attention_fwd(q, k, v, mask, scale, sink_omega):
    q_scaled = q * scale
    scores = jnp.matmul(q_scaled, jnp.swapaxes(k, -1, -2))

    s_sq = scores * scores
    one = jnp.array(1.0, dtype=scores.dtype)
    rad = one + s_sq
    r = lax.rsqrt(rad)
    u = scores * r
    denom = jnp.where(scores < 0, one - u, one)
    rho = jnp.where(scores < 0, r / denom, scores + rad * r)
    k2 = rho * rho
    k4 = k2 * k2
    k8 = k4 * k4

    p_masked = jnp.where(mask, k8, 0.0)
    d = jnp.sum(p_masked, axis=-1, keepdims=True) + jnp.array(sink_omega, dtype=v.dtype)
    w = p_masked * lax.reciprocal(d)
    out = jnp.matmul(w, v)

    return out, (q_scaled, k, v, w, r, scale, out)


def _causal_algebraic_attention_bwd(mask, scale, sink_omega, cache, g_out):
    del mask, sink_omega
    q_scaled, k, v, w, r, scale, out = cache

    # FlashAttention-style analytical scalar row reduction along feature dim D
    D_i = jnp.sum(g_out * out, axis=-1, keepdims=True)
    g_w = jnp.matmul(g_out, jnp.swapaxes(v, -1, -2))
    ds = (8.0 * scale) * r * w * (g_w - D_i)

    dq = jnp.matmul(ds, k)
    dk = jnp.matmul(jnp.swapaxes(ds, -1, -2), q_scaled)
    dv = jnp.matmul(jnp.swapaxes(w, -1, -2), g_out)

    return (dq.astype(q_scaled.dtype), dk.astype(k.dtype), dv.astype(v.dtype))


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _causal_algebraic_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    scale: float,
    sink_omega: float = 0.5,
) -> jax.Array:
    """Pure Algebraic Causal Attention with single-pass analytical FlashAttention VJP.

    Args:
        q: Query tensor (B, H, T, D)
        k: Key tensor (B, H, T, D)
        v: Value tensor (B, H, T, D)
        mask: Precomputed causal boolean mask (1, 1, T, T)
        scale: Precomputed attention scale factor
        sink_omega: Nonnegative attention sink mass Omega

    Returns:
        Attention output tensor (B, H, T, D)
    """
    return _causal_algebraic_attention_fwd(q, k, v, mask, scale, sink_omega)[0]


_causal_algebraic_attention.defvjp(_causal_algebraic_attention_fwd, _causal_algebraic_attention_bwd)


def _fused_oace_softmax_fwd(logits, targets, eps, gamma):
    inv_w = 1.0 / logits.shape[-1]
    tau = jax.lax.rsqrt(jnp.sum(logits.astype(jnp.float32) ** 2, axis=-1, keepdims=True) * inv_w + eps).astype(logits.dtype)
    normed = logits * tau

    s_sq = normed * normed
    one = jnp.array(1.0, dtype=normed.dtype)
    rad = one + s_sq
    r = jax.lax.rsqrt(rad)
    u = normed * r
    denom = jnp.where(normed < 0, one - u, one)
    rho = jnp.where(normed < 0, r / denom, normed + rad * r)

    k2 = rho * rho
    k4 = k2 * k2
    k8 = k4 * k4
    sum_k = jnp.sum(k8, axis=-1, keepdims=True)
    inv_S = jax.lax.reciprocal(sum_k)
    p = k8 * inv_S

    S_eighth = jax.lax.rsqrt(jax.lax.rsqrt(jax.lax.rsqrt(inv_S)))
    rho_7 = rho * k2 * k4
    p_78 = (S_eighth * inv_S) * rho_7
    sum_p78 = jnp.sum(p_78, axis=-1, keepdims=True)

    rho_c = jnp.take_along_axis(rho, targets[..., None], axis=-1)
    p_c_inv8 = S_eighth * jax.lax.reciprocal(rho_c)

    loss = gamma * (8.0 * p_c_inv8.squeeze(-1) + (8.0 / 7.0) * sum_p78.squeeze(-1) - (64.0 / 7.0))
    mean_loss = jnp.mean(loss)
    return mean_loss, (normed, tau, p, p_78, p_c_inv8, r, sum_p78, targets)


def _fused_oace_softmax_bwd(eps, gamma, cache, g):
    del eps
    normed, tau, p, p_78, p_c_inv8, r, sum_p78, targets = cache
    V = normed.shape[-1]
    scalar_diff = sum_p78 - p_c_inv8
    one_hot = jax.nn.one_hot(targets, V, dtype=normed.dtype)
    bracket = p_78 - one_hot * p_c_inv8 - p * scalar_diff
    g_y = (8.0 * gamma / float(targets.size)) * r * bracket * g
    inv_w = 1.0 / V
    radial = jnp.sum(g_y * normed, axis=-1, keepdims=True) * inv_w
    dx = tau * (g_y - radial * normed)
    return (dx.astype(normed.dtype), None)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def fused_oace_softmax_loss(logits, targets, eps=100.0, gamma=2.0):
    """Factored closed-simplex A-Softmax and strictly proper OACE loss functional with analytical VJP.

    Evaluates Bregman divergence L_{1/8} over closed vocabulary simplex with zero transcendentals.
    Factors p_i^{-1/8} = S^{1/8} / rho_i and p_i^{7/8} = (S^{1/8} / S) * rho_i^7, replacing
    3*V sequential rsqrt instructions on 50,257 elements with a single scalar rsqrt cascade
    on the partition sum S.
    """
    return _fused_oace_softmax_fwd(logits, targets, eps, gamma)[0]


fused_oace_softmax_loss.defvjp(_fused_oace_softmax_fwd, _fused_oace_softmax_bwd)


class AlgebraicTransformerLM:
    """Pure Algebraic Causal Transformer language model."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.head_dim = self.config.d_model // self.config.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for AGO Cayley rotation, got {self.head_dim}")

    def init_params(self, key: jax.Array) -> Dict[str, Any]:
        cfg = self.config
        keys = jax.random.split(key, 25)
        k_idx = 0

        def normal(shape, std=0.02):
            nonlocal k_idx
            k_idx += 1
            return (jax.random.normal(keys[k_idx % len(keys)], shape) * std).astype(cfg.param_dtype)

        params: Dict[str, Any] = {
            "token_embed": normal((cfg.vocab_size, cfg.d_model)),
            "layers": [],
        }

        if not cfg.tie_embeddings:
            params["output_head"] = normal((cfg.d_model, cfg.vocab_size))

        for _ in range(cfg.num_layers):
            layer = {
                "w_q": normal((cfg.d_model, cfg.d_model)),
                "w_k": normal((cfg.d_model, cfg.d_model)),
                "w_v": normal((cfg.d_model, cfg.d_model)),
                "w_o": normal((cfg.d_model, cfg.d_model)),
                "w_g": normal((cfg.d_model, cfg.d_ff)),
                "w_u": normal((cfg.d_model, cfg.d_ff)),
                "w_d": normal((cfg.d_ff, cfg.d_model)),
            }
            params["layers"].append(layer)

        return params

    def forward(
        self,
        params: Dict[str, Any],
        tokens: jax.Array,
        rotary_params: Optional[CayleyRotary] = None,
    ) -> jax.Array:
        """Forward pass of AlgebraicTransformerLM returning un-embedding logits."""
        cfg = self.config
        B, T = tokens.shape

        if rotary_params is None:
            rotary_params = build_cayley_rotary_matrix(self.head_dim, cfg.max_seq_len, dtype=cfg.dtype)

        scale = float(jax.lax.rsqrt(jnp.array(self.head_dim, dtype=jnp.float32)))
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]
        c_raw, s_raw = _extract_cs(rotary_params)
        c = _align_param(c_raw, (B, cfg.num_heads, T, self.head_dim), seq_axis=2)
        s = _align_param(s_raw, (B, cfg.num_heads, T, self.head_dim), seq_axis=2)

        # 1. Token Embedding + Parameter-Free AVN
        x = params["token_embed"][tokens].astype(cfg.dtype)
        x = avn(x, eps=cfg.eps)

        for layer in params["layers"]:
            # 2. Multi-Head Attention Sub-Layer
            h1 = avn(x, eps=cfg.eps)
            q = jnp.matmul(h1, layer["w_q"].astype(cfg.dtype)).reshape(B, T, cfg.num_heads, self.head_dim).swapaxes(1, 2)
            k = jnp.matmul(h1, layer["w_k"].astype(cfg.dtype)).reshape(B, T, cfg.num_heads, self.head_dim).swapaxes(1, 2)
            v = jnp.matmul(h1, layer["w_v"].astype(cfg.dtype)).reshape(B, T, cfg.num_heads, self.head_dim).swapaxes(1, 2)

            # Apply pre-aligned AGO Cayley Rotations
            q_rot = _rotate_tensor(q, c, s)
            k_rot = _rotate_tensor(k, c, s)

            # Octic AFA Causal Attention
            attn_out = _causal_algebraic_attention(q_rot, k_rot, v, mask, scale, sink_omega=cfg.sink_omega)
            attn_flat = attn_out.swapaxes(1, 2).reshape(B, T, cfg.d_model)
            x = x + jnp.matmul(attn_flat, layer["w_o"].astype(cfg.dtype))

            # 3. ALU-GLU FFN Sub-Layer
            h2 = avn(x, eps=cfg.eps)
            ffn_out = _alu_glu(
                h2,
                layer["w_g"].astype(cfg.dtype),
                layer["w_u"].astype(cfg.dtype),
                layer["w_d"].astype(cfg.dtype),
            )
            x = x + ffn_out

        # 4. Final Parameter-Free AVN
        x_final = avn(x, eps=cfg.eps)

        # 5. Linear Un-Embedding Head
        if cfg.tie_embeddings:
            logits = jnp.matmul(x_final, params["token_embed"].T.astype(cfg.dtype))
        else:
            logits = jnp.matmul(x_final, params["output_head"].astype(cfg.dtype))

        return logits

    def loss(
        self,
        params: Dict[str, Any],
        tokens: jax.Array,
        targets: jax.Array,
        rotary_params: Optional[CayleyRotary] = None,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        """Computes OACE loss functional over token sequence."""
        cfg = self.config
        logits = self.forward(params, tokens, rotary_params=rotary_params)

        loss = fused_oace_softmax_loss(logits, targets, eps=cfg.eps_vocab, gamma=cfg.gamma)
        return loss, {"logits": logits, "loss": loss}
