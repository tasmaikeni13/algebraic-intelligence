"""Phase 8 hyperparameter sweep experimental logic and training configurations.

Implements the equal-budget search space and step functions for 125M scale models
on FineWeb-Edu across Seeds 42, 43, 44.
"""

from dataclasses import dataclass, asdict
from functools import partial
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from src.model import AlgebraicTransformerLM, ModelConfig, count_parameters
from src.baseline import StandardTransformerLM, BaselineConfig
from src.optimizer import algebraic_adamw, ards_schedule
from src.attention import build_cayley_rotary_matrix
from src.baseline import _build_standard_rope
from scripts.audit_primitives import source_audit

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class HparamConfig:
    """Hyperparameter bundle for a candidate run."""
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    sink_omega: float = 0.5
    gamma: float = 2.0
    min_lr: float = 1e-5
    schedule: str = "ards"  # "ards" or "cosine"
    max_grad_norm: float = 1.0


def get_125m_algebraic_config(sink_omega: float = 0.5, gamma: float = 2.0) -> ModelConfig:
    """Returns 125M parameter ModelConfig for AlgebraicTransformerLM."""
    return ModelConfig(
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=2048,
        max_seq_len=2048,
        eps=1e-5,
        eps_vocab=100.0,
        sink_omega=sink_omega,
        gamma=gamma,
        tie_embeddings=True,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    )


def get_125m_baseline_config() -> BaselineConfig:
    """Returns 125M parameter BaselineConfig for StandardTransformerLM."""
    return BaselineConfig(
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=2048,
        max_seq_len=2048,
        eps=1e-5,
        tie_embeddings=True,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    )


def _clip_grad_norm_algebraic(grads: Any, max_norm: float = 1.0) -> Tuple[Any, jax.Array]:
    """Clips gradients using pure hardware rsqrt without transcendental functions."""
    leaves = jax.tree_util.tree_leaves(grads)
    sum_sq = sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in leaves)
    safe_sq = jnp.maximum(sum_sq, 1e-12)
    inv_norm = lax.rsqrt(safe_sq)
    norm = sum_sq * inv_norm

    clip_factor = jnp.minimum(1.0, max_norm * inv_norm)
    clipped_grads = jax.tree_util.tree_map(lambda g: (g * clip_factor).astype(g.dtype), grads)
    clipped_norm = norm * clip_factor
    return clipped_grads, clipped_norm


def create_cosine_schedule(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-5,
) -> Callable[[int], jax.Array]:
    """Standard Cosine Annealing learning rate schedule for baseline model."""
    def schedule(step):
        step_f = jnp.asarray(step, dtype=jnp.float32)
        warmup_f = float(warmup_steps)
        total_f = float(total_steps)

        warmup_factor = jnp.minimum(1.0, step_f / jnp.maximum(1.0, warmup_f))
        progress = jnp.maximum(0.0, (step_f - warmup_f) / jnp.maximum(1.0, total_f - warmup_f))
        cosine_decay = 0.5 * (1.0 + jnp.cos(math.pi * jnp.minimum(1.0, progress)))
        lr = min_lr + (learning_rate - min_lr) * cosine_decay
        return jnp.where(step_f < warmup_f, learning_rate * warmup_factor, lr)

    return schedule


def train_step_algebraic_fn(
    model: AlgebraicTransformerLM,
    optimizer_tx,
    rotary_params,
    max_grad_norm: float = 1.0,
):
    """Factory returning a JIT-compilable single training step for AlgebraicTransformerLM."""
    def step_fn(params, opt_state, tokens, targets):
        def loss_fn(p):
            loss_val, aux = model.loss(p, tokens, targets, rotary_params=rotary_params)
            return loss_val, aux

        (loss, aux), raw_grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        clipped_grads, grad_norm = _clip_grad_norm_algebraic(raw_grads, max_grad_norm)
        updates, new_opt_state = optimizer_tx.update(clipped_grads, opt_state, params)
        new_params = jax.tree_util.tree_map(lambda p, u: (p + u).astype(p.dtype), params, updates)

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "is_finite": jnp.isfinite(loss) & jnp.isfinite(grad_norm),
        }
        return new_params, new_opt_state, metrics

    return step_fn


def train_step_baseline_fn(
    model: StandardTransformerLM,
    optimizer_tx,
    cos_angles,
    sin_angles,
    max_grad_norm: float = 1.0,
):
    """Factory returning a JIT-compilable single training step for StandardTransformerLM."""
    def step_fn(params, opt_state, tokens, targets):
        def loss_fn(p):
            loss_val, aux = model.loss(p, tokens, targets, cos_angles=cos_angles, sin_angles=sin_angles)
            return loss_val, aux

        (loss, aux), raw_grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        clipped_grads, grad_norm = _clip_grad_norm_algebraic(raw_grads, max_grad_norm)
        updates, new_opt_state = optimizer_tx.update(clipped_grads, opt_state, params)
        new_params = jax.tree_util.tree_map(lambda p, u: (p + u).astype(p.dtype), params, updates)

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "is_finite": jnp.isfinite(loss) & jnp.isfinite(grad_norm),
        }
        return new_params, new_opt_state, metrics

    return step_fn


def evaluate_perplexity_fast(
    model,
    params,
    valid_tokens: np.ndarray,
    seq_len: int = 2048,
    batch_size: int = 4,
    num_eval_batches: int = 10,
    is_algebraic: bool = True,
    rotary_or_angles: Optional[Any] = None,
) -> Tuple[float, float]:
    """Fast validation perplexity and loss evaluation on FineWeb-Edu.

    Returns:
        (perplexity, average_loss)
    """
    total_tokens = len(valid_tokens)
    span = seq_len + 1
    total_nll = 0.0
    total_count = 0

    max_b = min(num_eval_batches, (total_tokens - span) // (batch_size * seq_len))

    for b in range(max(1, max_b)):
        starts = [b * batch_size * seq_len + i * seq_len for i in range(batch_size)]
        x = np.stack([valid_tokens[s : s + seq_len].astype(np.int32) for s in starts])
        y = np.stack([valid_tokens[s + 1 : s + span].astype(np.int32) for s in starts])

        x_j = jnp.asarray(x)
        y_j = jnp.asarray(y)

        if is_algebraic:
            rotary = rotary_or_angles
            if rotary is None:
                rotary = build_cayley_rotary_matrix(model.head_dim, seq_len)
            logits = model.forward(params, x_j, rotary_params=rotary)
            from src.attention import algebraic_softmax
            eps_v = float(getattr(model.config, "eps_vocab", 100.0))
            probs = algebraic_softmax(logits, sink_omega=0.0, eps=eps_v)
            probs = jnp.maximum(probs, 1e-12)
            target_probs = jnp.take_along_axis(probs, y_j[..., None], axis=-1).squeeze(-1)
            batch_nll = float(jax.device_get(-jnp.sum(jnp.log(target_probs))))
        else:
            if rotary_or_angles is None:
                cos_angles, sin_angles = _build_standard_rope(model.head_dim, seq_len)
            else:
                cos_angles, sin_angles = rotary_or_angles
            logits = model.forward(params, x_j, cos_angles=cos_angles, sin_angles=sin_angles)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(log_probs, y_j[..., None], axis=-1).squeeze(-1)
            batch_nll = float(jax.device_get(-jnp.sum(target_log_probs)))

        total_nll += batch_nll
        total_count += x.size

    avg_nll = total_nll / total_count
    ppl = float(np.exp(min(avg_nll, 20.0)))
    return ppl, avg_nll
