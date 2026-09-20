"""Standard Causal Transformer baseline (Transcendental reference model).

Implements the standard Transformer architecture for comparative benchmarking:
- SwiGLU feed-forward network with x * sigmoid(x) = x / (1 + exp(-x))
- RMSNorm with learnable gamma scale parameters
- Standard exponential Softmax attention: exp(s) / sum(exp(s))
- Standard RoPE rotary positional encodings: (cos(m*theta), sin(m*theta))
- Cross-Entropy loss functional: -ln(p_k)
- Standard AdamW optimizer with Cosine Annealing learning rate schedule
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class BaselineConfig:
    vocab_size: int = 50257
    d_model: int = 288
    num_layers: int = 6
    num_heads: int = 6
    d_ff: int = 768
    max_seq_len: int = 512
    eps: float = 1e-5
    tie_embeddings: bool = True
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32
    remat: bool = False


def _standard_rmsnorm(x: jax.Array, gamma: jax.Array, eps: float = 1e-5) -> jax.Array:
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    return (normed * gamma).astype(x.dtype)


def _build_standard_rope(dim: int, max_seq_len: int, base: float = 10000.0) -> Tuple[jax.Array, jax.Array]:
    half_dim = dim // 2
    theta = 1.0 / (base ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    pos = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, theta)
    cos_angles = jnp.cos(angles)
    sin_angles = jnp.sin(angles)
    return cos_angles, sin_angles


def _apply_standard_rope(
    q: jax.Array,
    k: jax.Array,
    cos_angles: jax.Array,
    sin_angles: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    seq_len = q.shape[2]
    cos_t = cos_angles[:seq_len][None, None, :, :]
    sin_t = sin_angles[:seq_len][None, None, :, :]

    def rotate_half(x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return jnp.concatenate([-x2, x1], axis=-1)

    cos_full = jnp.concatenate([cos_t, cos_t], axis=-1).astype(q.dtype)
    sin_full = jnp.concatenate([sin_t, sin_t], axis=-1).astype(q.dtype)

    q_rot = q * cos_full + rotate_half(q) * sin_full
    k_rot = k * cos_full + rotate_half(k) * sin_full
    return q_rot, k_rot


def _standard_swiglu(x: jax.Array, w_g: jax.Array, w_u: jax.Array, w_d: jax.Array) -> jax.Array:
    gate = jnp.matmul(x, w_g)
    up = jnp.matmul(x, w_u)
    # Swish: z * sigmoid(z)
    swish_up = up * jax.nn.sigmoid(up)
    return jnp.matmul(gate * swish_up, w_d)


def _baseline_layer_forward(x, layer, cos_angles, sin_angles, causal_mask):
    B, T, d_model = x.shape
    dtype = x.dtype
    head_dim = cos_angles.shape[-1] * 2
    num_heads = d_model // head_dim
    scale = 1.0 / math.sqrt(head_dim)
    eps = 1e-5

    h = _standard_rmsnorm(x, layer["norm1_gamma"], eps)
    q = jnp.matmul(h, layer["w_q"].astype(dtype)).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)
    k = jnp.matmul(h, layer["w_k"].astype(dtype)).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)
    v = jnp.matmul(h, layer["w_v"].astype(dtype)).reshape(B, T, num_heads, head_dim).swapaxes(1, 2)

    q_rot, k_rot = _apply_standard_rope(q, k, cos_angles, sin_angles)

    scores = jnp.matmul(q_rot, k_rot.swapaxes(-1, -2)) * scale
    scores = jnp.where(causal_mask, scores, -1e4)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.matmul(attn_weights, v).swapaxes(1, 2).reshape(B, T, d_model)
    x = x + jnp.matmul(attn_out, layer["w_o"].astype(dtype))

    h2 = _standard_rmsnorm(x, layer["norm2_gamma"], eps)
    ffn_out = _standard_swiglu(
        h2,
        layer["w_g"].astype(dtype),
        layer["w_u"].astype(dtype),
        layer["w_d"].astype(dtype),
    )
    return x + ffn_out


class StandardTransformerLM:
    """Standard Causal Transformer baseline."""

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        self.head_dim = self.config.d_model // self.config.num_heads

    def init_params(self, key: jax.Array) -> Dict[str, Any]:
        cfg = self.config
        keys = jax.random.split(key, 20)
        k_idx = 0

        def normal(shape, std=0.02):
            nonlocal k_idx
            k_idx += 1
            return (jax.random.normal(keys[k_idx % len(keys)], shape) * std).astype(cfg.param_dtype)

        params: Dict[str, Any] = {
            "token_embed": normal((cfg.vocab_size, cfg.d_model)),
            "embed_norm_gamma": jnp.ones(cfg.d_model, dtype=cfg.param_dtype),
            "final_norm_gamma": jnp.ones(cfg.d_model, dtype=cfg.param_dtype),
            "layers": [],
        }

        if not cfg.tie_embeddings:
            params["output_head"] = normal((cfg.d_model, cfg.vocab_size))

        for _ in range(cfg.num_layers):
            layer = {
                "norm1_gamma": jnp.ones(cfg.d_model, dtype=cfg.param_dtype),
                "w_q": normal((cfg.d_model, cfg.d_model)),
                "w_k": normal((cfg.d_model, cfg.d_model)),
                "w_v": normal((cfg.d_model, cfg.d_model)),
                "w_o": normal((cfg.d_model, cfg.d_model)),
                "norm2_gamma": jnp.ones(cfg.d_model, dtype=cfg.param_dtype),
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
        cos_angles: Optional[jax.Array] = None,
        sin_angles: Optional[jax.Array] = None,
    ) -> jax.Array:
        cfg = self.config
        B, T = tokens.shape

        if cos_angles is None or sin_angles is None:
            cos_angles, sin_angles = _build_standard_rope(self.head_dim, cfg.max_seq_len)

        x = params["token_embed"][tokens].astype(cfg.dtype)
        x = _standard_rmsnorm(x, params["embed_norm_gamma"], cfg.eps)

        scale = 1.0 / math.sqrt(self.head_dim)
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]

        layer_fn = jax.checkpoint(_baseline_layer_forward) if cfg.remat else _baseline_layer_forward
        for layer in params["layers"]:
            x = layer_fn(x, layer, cos_angles, sin_angles, causal_mask)

        x_final = _standard_rmsnorm(x, params["final_norm_gamma"], cfg.eps)

        if cfg.tie_embeddings:
            logits = jnp.matmul(x_final, params["token_embed"].T.astype(cfg.dtype))
        else:
            logits = jnp.matmul(x_final, params["output_head"].astype(cfg.dtype))

        return logits.astype(jnp.float32)

    def loss(
        self,
        params: Dict[str, Any],
        tokens: jax.Array,
        targets: jax.Array,
        cos_angles: Optional[jax.Array] = None,
        sin_angles: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        logits = self.forward(params, tokens, cos_angles, sin_angles)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
        loss = -jnp.mean(target_log_probs)
        return loss, {"loss": loss}
