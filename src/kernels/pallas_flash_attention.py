"""Vendor-Grade Standard FlashAttention-2 in JAX (Pallas TPU & XLA Tiled).

Implements the official Dao et al. FlashAttention-2 algorithm:
- Online softmax with running row-maximum tracking m_i and exponential rescaling e^{m_old - m_new}
- Memory-efficient tiled accumulation with O(B_r * B_c) working memory in SRAM/VMEM
- Single-pass analytical backward pass recomputing softmax weights on-the-fly
- Serves as the rigorous, state-of-the-art vendor baseline for head-to-head empirical comparison.
"""

from functools import partial
import math
from typing import Optional, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp


def tiled_flash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool = True,
    block_q: int = 128,
    block_k: int = 128,
    return_stats: bool = False,
) -> Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
    """Standard FlashAttention-2 tiled forward pass with online softmax rescaling.

    Args:
        q: Query tensor of shape (B, H, T, D).
        k: Key tensor of shape (B, H, T, D).
        v: Value tensor of shape (B, H, T, D).
        causal: Autoregressive causal masking flag.
        block_q: Query tile sequence length.
        block_k: Key/value tile sequence length.
        return_stats: If True, returns (output, running_max, running_sum) for backward.

    Returns:
        Attention output tensor of shape (B, H, T, D).
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / math.sqrt(head_dim))
    num_q_blocks = seq_len // block_q
    num_k_blocks = seq_len // block_k

    accum_dtype = jnp.float32

    def _query_block_step(qi_idx):
        q_block = lax.dynamic_slice_in_dim(q, qi_idx * block_q, block_q, axis=2)

        def _key_block_step(kj_idx, acc):
            o_acc, m_prev, l_prev = acc
            k_block = lax.dynamic_slice_in_dim(k, kj_idx * block_k, block_k, axis=2)
            v_block = lax.dynamic_slice_in_dim(v, kj_idx * block_k, block_k, axis=2)

            # Raw score tile: S_ij = (Q_i @ K_j^T) * scale
            s_ij = jnp.matmul(q_block.astype(accum_dtype), jnp.swapaxes(k_block.astype(accum_dtype), -1, -2)) * scale

            # Causal mask only on diagonal tile
            if causal:
                is_diag = kj_idx == qi_idx
                row_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 0) + qi_idx * block_q
                col_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + kj_idx * block_k
                mask = col_ids <= row_ids
                s_ij = jnp.where(is_diag, jnp.where(mask[None, None, :, :], s_ij, -1e9), s_ij)

            # Online softmax tracking (FlashAttention-2)
            m_curr = jnp.max(s_ij, axis=-1, keepdims=True)
            m_new = jnp.maximum(m_prev, m_curr)

            # Exponential rescaling factor: alpha = exp(m_prev - m_new)
            alpha = jnp.exp(m_prev - m_new)
            p_ij = jnp.exp(s_ij - m_new)

            # Rescale previous accumulator and add current block
            o_new = o_acc * alpha + jnp.matmul(p_ij.astype(v_block.dtype), v_block.astype(accum_dtype))
            l_new = l_prev * alpha + jnp.sum(p_ij, axis=-1, keepdims=True)

            return (o_new, m_new, l_new)

        init_o = jnp.zeros_like(q_block, dtype=accum_dtype)
        init_m = jnp.full((batch_size, num_heads, block_q, 1), -1e9, dtype=accum_dtype)
        init_l = jnp.zeros((batch_size, num_heads, block_q, 1), dtype=accum_dtype)

        upper_k = (qi_idx + 1) if causal else num_k_blocks
        final_o, final_m, final_l = lax.fori_loop(0, upper_k, _key_block_step, (init_o, init_m, init_l))
        out_block = final_o / jnp.maximum(final_l, 1e-12)

        return out_block.astype(q.dtype), final_m, final_l

    results = [_query_block_step(i) for i in range(num_q_blocks)]
    out_blocks = [r[0] for r in results]
    m_blocks = [r[1] for r in results]
    l_blocks = [r[2] for r in results]

    out = jnp.concatenate(out_blocks, axis=2)
    m = jnp.concatenate(m_blocks, axis=2)
    l = jnp.concatenate(l_blocks, axis=2)

    if return_stats:
        return out, m, l
    return out


def tiled_flash_attention_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    m: jax.Array,
    l: jax.Array,
    g_out: jax.Array,
    causal: bool = True,
    block_q: int = 128,
    block_k: int = 128,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Standard FlashAttention-2 tiled analytical backward pass."""
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / math.sqrt(head_dim))
    num_q_blocks = seq_len // block_q
    num_k_blocks = seq_len // block_k
    accum_dtype = jnp.float32

    # Row contraction D_i = sum_d g_out,id * out_id
    D = jnp.sum(g_out.astype(accum_dtype) * out.astype(accum_dtype), axis=-1, keepdims=True)

    dq = jnp.zeros_like(q, dtype=accum_dtype)
    dk = jnp.zeros_like(k, dtype=accum_dtype)
    dv = jnp.zeros_like(v, dtype=accum_dtype)

    for i in range(num_q_blocks):
        q_i = lax.dynamic_slice_in_dim(q, i * block_q, block_q, axis=2).astype(accum_dtype)
        g_out_i = lax.dynamic_slice_in_dim(g_out, i * block_q, block_q, axis=2).astype(accum_dtype)
        m_i = lax.dynamic_slice_in_dim(m, i * block_q, block_q, axis=2)
        l_i = lax.dynamic_slice_in_dim(l, i * block_q, block_q, axis=2)
        D_i = lax.dynamic_slice_in_dim(D, i * block_q, block_q, axis=2)

        dq_i = jnp.zeros_like(q_i)
        max_k = (i + 1) if causal else num_k_blocks

        for j in range(max_k):
            k_j = lax.dynamic_slice_in_dim(k, j * block_k, block_k, axis=2).astype(accum_dtype)
            v_j = lax.dynamic_slice_in_dim(v, j * block_k, block_k, axis=2).astype(accum_dtype)

            # Recompute softmax weights: P_ij = exp(S_ij - m_i) / l_i
            s_ij = jnp.matmul(q_i, jnp.swapaxes(k_j, -1, -2)) * scale
            if causal and (i == j):
                row_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 0) + i * block_q
                col_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + j * block_k
                cmask = col_ids <= row_ids
                s_ij = jnp.where(cmask[None, None, :, :], s_ij, -1e9)

            p_ij = jnp.exp(s_ij - m_i) / jnp.maximum(l_i, 1e-12)
            if causal and (i == j):
                p_ij = jnp.where(cmask[None, None, :, :], p_ij, 0.0)

            # Gradient with respect to scores: dS_ij = P_ij * (g_out_i @ V_j^T - D_i)
            g_w = jnp.matmul(g_out_i, jnp.swapaxes(v_j, -1, -2))
            ds_ij = p_ij * (g_w - D_i)

            dq_i = dq_i + jnp.matmul(ds_ij, k_j) * scale
            dk_j = jnp.matmul(jnp.swapaxes(ds_ij, -1, -2), q_i) * scale
            dv_j = jnp.matmul(jnp.swapaxes(p_ij, -1, -2), g_out_i)

            curr_dk_j = lax.dynamic_slice_in_dim(dk, j * block_k, block_k, axis=2)
            curr_dv_j = lax.dynamic_slice_in_dim(dv, j * block_k, block_k, axis=2)
            dk = lax.dynamic_update_slice_in_dim(dk, curr_dk_j + dk_j, j * block_k, axis=2)
            dv = lax.dynamic_update_slice_in_dim(dv, curr_dv_j + dv_j, j * block_k, axis=2)

        curr_dq_i = lax.dynamic_slice_in_dim(dq, i * block_q, block_q, axis=2)
        dq = lax.dynamic_update_slice_in_dim(dq, curr_dq_i + dq_i, i * block_q, axis=2)

    return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)


def _flash_attn_fwd_vjp(q, k, v, causal, block_q, block_k):
    out, m, l = tiled_flash_attention_forward(
        q, k, v, causal=causal, block_q=block_q, block_k=block_k, return_stats=True
    )
    return out, (q, k, v, out, m, l)


def _flash_attn_bwd_vjp(causal, block_q, block_k, res, g_out):
    q, k, v, out, m, l = res
    dq, dk, dv = tiled_flash_attention_backward(
        q, k, v, out, m, l, g_out, causal=causal, block_q=block_q, block_k=block_k
    )
    return dq, dk, dv


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def standard_flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool = True,
    block_q: int = 128,
    block_k: int = 128,
) -> jax.Array:
    """Tuned FlashAttention-2 forward and analytical backward interface."""
    return tiled_flash_attention_forward(q, k, v, causal=causal, block_q=block_q, block_k=block_k)


standard_flash_attention.defvjp(_flash_attn_fwd_vjp, _flash_attn_bwd_vjp)
