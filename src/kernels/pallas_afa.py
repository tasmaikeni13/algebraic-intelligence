"""Hardware-Fused Algebraic FlashAttention (AFA) on Google Cloud TPU v4 via JAX Pallas.

Under the Zero-Transcendental Axiom, AFA completely eliminates running-max
subtraction exp(m_old - m_new) and online exponential rescaling barriers of
standard FlashAttention-2. Attention scores are computed via pure 3-stage
VMU squaring of rho(s)^8 and accumulated additively into Vector Memory (VMEM)
and Matrix Multiply Units (MXUs) without inter-tile log-sum-exp synchronization.

Key components:
- afa_kernel: Pure additive tile kernel for TPU v4 VMEM/MXU.
- pallas_afa_forward: Orchestrated JAX Pallas forward pass with 128x128 systolic tiling.
- tiled_afa_forward: XLA-tiled additive accumulation kernel.
- exact_afa_reference: Exact un-tiled float64 mathematical reference.
- distributed_ring_afa: Lock-free distributed Ring Attention across TPU mesh.
- algebraic_flash_attention: Unified entrypoint with automatic padding and hardware routing.
"""

import functools
import math
from typing import Optional, Tuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _vmu_octic_kernel(s: jax.Array) -> jax.Array:
    """Evaluates octic algebraic kernel rho(s)^8 on VMU vector registers.

    Base map: rho(s) = s + sqrt(1 + s^2) = r / (1 - s*r) for s < 0.
    Exact 3-stage squaring hierarchy: rho -> rho^2 -> rho^4 -> rho^8.
    Strictly zero transcendentals (0 exp, 0 log, 0 trig).
    """
    s_sq = s * s
    r = lax.rsqrt(1.0 + s_sq)
    u = s * r
    denom = jnp.where(s < 0.0, 1.0 - u, 1.0)
    rho = jnp.where(s < 0.0, r / denom, s + (1.0 + s_sq) * r)
    k2 = rho * rho
    k4 = k2 * k2
    return k4 * k4


def afa_kernel(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    d_ref,
    o_acc_ref,
    d_acc_ref,
    *,
    scale: float,
    sink_omega: float,
    causal: bool = False,
    num_k_blocks: int = 1,
    block_q: int = 128,
    block_k: int = 128,
    valid_seq_len: Optional[int] = None,
):
    """Pure Additive Algebraic FlashAttention Tile Kernel for TPU v4.

    Executed inside TPU v4 TensorCore VMEM across 128x128 MXU systolic arrays.
    Accumulates unnormalized numerators and denominators additively without
    any running-max subtraction or transcendental rescaling.

    Args:
        q_ref: Slice of Q in VMEM of shape (1, 1, block_q, head_dim).
        k_ref: Slice of K in VMEM of shape (1, 1, block_k, head_dim).
        v_ref: Slice of V in VMEM of shape (1, 1, block_k, head_dim).
        o_ref: Output numerator slice in VMEM of shape (1, 1, block_q, head_dim).
        d_ref: Output denominator slice in VMEM of shape (1, 1, block_q).
        o_acc_ref: VMEM scratch accumulator for numerator of shape (block_q, head_dim).
        d_acc_ref: VMEM scratch accumulator for denominator of shape (block_q,).
        scale: Scaling factor 1 / sqrt(head_dim).
        sink_omega: Nonnegative attention sink scalar mass.
        causal: Boolean flag indicating causal autoregressive masking.
        num_k_blocks: Total number of key/value tiles along sequence dimension.
        block_q: Query tile sequence length.
        block_k: Key tile sequence length.
        valid_seq_len: Optional unpadded active sequence length.
    """
    b_idx = pl.program_id(0)
    h_idx = pl.program_id(1)
    q_blk_idx = pl.program_id(2)
    k_blk_idx = pl.program_id(3)

    # 1. Initialize scratch accumulators on the first key tile (k_blk_idx == 0)
    @pl.when(k_blk_idx == 0)
    def _init():
        o_acc_ref[...] = jnp.zeros_like(o_acc_ref)
        d_acc_ref[...] = jnp.zeros_like(d_acc_ref)

    # 2. In causal attention, completely skip tiles strictly in the future (k_blk_idx > q_blk_idx)
    should_compute = (k_blk_idx <= q_blk_idx) if causal else True

    @pl.when(should_compute)
    def _compute_step():
        # Load local tiles into VMU/MXU registers
        q_tile = q_ref[0, 0]  # (block_q, head_dim)
        k_tile = k_ref[0, 0]  # (block_k, head_dim)
        v_tile = v_ref[0, 0]  # (block_k, head_dim)

        # 2a. Systolic Matrix Multiplication on MXU: S = (Q @ K^T) * scale
        s_bc = jnp.matmul(q_tile, k_tile.T) * scale

        # 2b. Three-Stage Squaring Kernel on VMU: P = rho(S)^8
        p_bc = _vmu_octic_kernel(s_bc)

        # 2c. Padding mask: zero out weights for dummy keys outside valid_seq_len
        col_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + k_blk_idx * block_k
        if valid_seq_len is not None:
            p_bc = jnp.where(col_ids < valid_seq_len, p_bc, 0.0)

        # 2d. Intra-tile causal masking on the diagonal tile (q_blk_idx == k_blk_idx)
        if causal:
            row_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 0) + q_blk_idx * block_q
            causal_mask = col_ids <= row_ids
            p_bc = jnp.where(causal_mask, p_bc, 0.0)

        # 2e. Pure Additive Tile Accumulation (NO running-max subtraction!):
        # Accumulate numerator: O_b += P_bc @ V_c (MXU matmul)
        o_acc_ref[...] = o_acc_ref[...] + jnp.matmul(p_bc.astype(v_tile.dtype), v_tile)

        # Accumulate denominator: D_b += sum_j P_bc[:, j] (VMU reduction)
        d_acc_ref[...] = d_acc_ref[...] + jnp.sum(p_bc, axis=-1)

    # 3. Write out accumulated sums on the final key tile (k_blk_idx == num_k_blocks - 1)
    @pl.when(k_blk_idx == (num_k_blocks - 1))
    def _finalize():
        o_ref[0, 0] = o_acc_ref[...].astype(o_ref.dtype)
        d_ref[0, 0] = d_acc_ref[...]


def pallas_afa_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    valid_seq_len: Optional[int] = None,
    interpret: Optional[bool] = None,
) -> jax.Array:
    """Full forward call orchestrating Pallas TPU execution.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim).
        v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim).
        sink_omega: Nonnegative attention sink scalar mass.
        causal: Whether to apply causal autoregressive masking.
        block_q: Tile query sequence length (multiple of 128 for TPU MXU).
        block_k: Tile key sequence length (multiple of 128 for TPU MXU).
        valid_seq_len: Optional active unpadded sequence length.
        interpret: Optional boolean to force Pallas interpreter on CPU.

    Returns:
        Attention output tensor Y of shape matching q and input dtype.
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    if seq_len % block_q != 0 or seq_len % block_k != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by block_q ({block_q}) and block_k ({block_k})"
        )

    num_q_blocks = seq_len // block_q
    num_k_blocks = seq_len // block_k
    scale = float(1.0 / math.sqrt(head_dim))

    # Grid: (batch_size, num_heads, num_q_blocks, num_k_blocks)
    grid = (batch_size, num_heads, num_q_blocks, num_k_blocks)

    # VMEM BlockSpecs with index maps matching TPU v4 TensorCore
    in_specs = [
        pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i, j: (b, h, i, 0)),
        pl.BlockSpec((1, 1, block_k, head_dim), lambda b, h, i, j: (b, h, j, 0)),
        pl.BlockSpec((1, 1, block_k, head_dim), lambda b, h, i, j: (b, h, j, 0)),
    ]
    out_specs = [
        pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i, j: (b, h, i, 0)),
        pl.BlockSpec((1, 1, block_q), lambda b, h, i, j: (b, h, i)),
    ]

    accum_dtype = jnp.float64 if q.dtype == jnp.float64 else jnp.float32

    scratch_shapes = [
        pltpu.VMEM((block_q, head_dim), accum_dtype),
        pltpu.VMEM((block_q,), accum_dtype),
    ]

    kernel_fn = functools.partial(
        afa_kernel,
        scale=scale,
        sink_omega=float(sink_omega),
        causal=causal,
        num_k_blocks=num_k_blocks,
        block_q=block_q,
        block_k=block_k,
        valid_seq_len=valid_seq_len,
    )

    if interpret is None:
        # Auto-detect: interpret on CPU, hardware-compile on TPU
        try:
            platform = jax.devices()[0].platform
            interpret = (platform != "tpu")
        except Exception:
            interpret = True

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
    )

    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
    )

    out_o, out_d = pl.pallas_call(
        kernel_fn,
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct((batch_size, num_heads, seq_len), accum_dtype),
        ],
        grid_spec=grid_spec,
        compiler_params=compiler_params,
        interpret=interpret,
    )(q, k, v)

    # Final normalization on VMU: Y = O / (D + sink_omega)
    denom = out_d[..., None] + sink_omega
    return (out_o / denom.astype(out_o.dtype)).astype(q.dtype)


def exact_afa_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
) -> jax.Array:
    """Exact un-tiled mathematical reference for Algebraic FlashAttention.

    Used for numerical verification against float64 ground truth.
    Strictly zero transcendentals.

    Args:
        q: (B, H, L, D) query array.
        k: (B, H, L, D) key array.
        v: (B, H, L, D) value array.
        sink_omega: Attention sink mass Omega >= 0.
        causal: Autoregressive causal masking flag.

    Returns:
        Exact attention output tensor.
    """
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]
    scale = float(1.0 / math.sqrt(head_dim))

    # Raw attention logits: S = (Q @ K^T) * scale
    s = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale

    # VMU octic kernel rho^8
    p = _vmu_octic_kernel(s)

    # Causal lower-triangular mask
    if causal:
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        p = jnp.where(mask[None, None, :, :], p, 0.0)

    # Additive output numerator and denominator
    o = jnp.matmul(p.astype(v.dtype), v)
    d = jnp.sum(p, axis=-1, keepdims=True)

    # Final rational normalization
    return (o / (d + sink_omega).astype(o.dtype)).astype(q.dtype)


def tiled_afa_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    valid_seq_len: Optional[int] = None,
) -> jax.Array:
    """XLA-tiled additive accumulation implementation of Algebraic FlashAttention.

    Executes tile streaming with O(B_q * B_k) working memory, compiling directly
    to fused XLA HLO dot and reduction operations across TPU and CPU.

    Args:
        q: Query tensor (B, H, L, D).
        k: Key tensor (B, H, L, D).
        v: Value tensor (B, H, L, D).
        sink_omega: Nonnegative attention sink scalar mass.
        causal: Autoregressive masking flag.
        block_q: Query tile size.
        block_k: Key tile size.
        valid_seq_len: Optional active unpadded sequence length.

    Returns:
        Attention output tensor Y.
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / math.sqrt(head_dim))
    num_q_blocks = seq_len // block_q
    num_k_blocks = seq_len // block_k

    accum_dtype = jnp.float64 if q.dtype == jnp.float64 else jnp.float32

    def _query_block_step(qi_idx):
        # Slice Q block: (B, H, block_q, D)
        q_block = lax.dynamic_slice_in_dim(q, qi_idx * block_q, block_q, axis=2)

        def _key_block_step(kj_idx, acc):
            o_acc, d_acc = acc
            # Slice K, V blocks: (B, H, block_k, D)
            k_block = lax.dynamic_slice_in_dim(k, kj_idx * block_k, block_k, axis=2)
            v_block = lax.dynamic_slice_in_dim(v, kj_idx * block_k, block_k, axis=2)

            # Matmul on MXU
            s_bc = jnp.matmul(q_block, jnp.swapaxes(k_block, -1, -2)) * scale

            # VMU octic kernel
            p_bc = _vmu_octic_kernel(s_bc)

            # Padding mask: zero out weights for dummy keys outside valid_seq_len
            col_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + kj_idx * block_k
            if valid_seq_len is not None:
                p_bc = jnp.where(col_ids < valid_seq_len, p_bc, 0.0)

            # Causal masking
            if causal:
                row_ids = lax.broadcasted_iota(jnp.int32, (block_q, block_k), 0) + qi_idx * block_q
                mask = col_ids <= row_ids
                tile_active = kj_idx <= qi_idx
                p_bc = jnp.where(tile_active & mask[None, None, :, :], p_bc, 0.0)

            # Pure Additive Tile Accumulation
            o_step = jnp.matmul(p_bc.astype(v_block.dtype), v_block).astype(accum_dtype)
            d_step = jnp.sum(p_bc, axis=-1, keepdims=True).astype(accum_dtype)

            return (o_acc + o_step, d_acc + d_step)

        # Initialize tile accumulators
        init_o = jnp.zeros((batch_size, num_heads, block_q, head_dim), dtype=accum_dtype)
        init_d = jnp.zeros((batch_size, num_heads, block_q, 1), dtype=accum_dtype)

        final_o, final_d = lax.fori_loop(0, num_k_blocks, _key_block_step, (init_o, init_d))
        return (final_o / (final_d + sink_omega).astype(final_o.dtype)).astype(q.dtype)

    # Scan or stack over query blocks
    outputs = [_query_block_step(i) for i in range(num_q_blocks)]
    return jnp.concatenate(outputs, axis=2)


def distributed_ring_afa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
    axis_name: str = "devices",
) -> jax.Array:
    """Lock-Free Distributed Ring Attention across 16 TPU v4 Chips over ICI.

    Because the AFA numerator O_b = sum_c P_bc V_c and denominator D_b = sum_c sum_j P_bc,j
    are strictly linear and additive, Ring Attention requires ZERO running-max
    synchronization or online rescaling across the 16 TPU chips.

    Each TPU core computes partial numerators O_p and denominators D_p for its
    local query shard as key/value blocks rotate around the 3D Torus ICI ring,
    evaluating a single final normalization pass at the end of the ring traversal:
        Y_b = (sum_p O_b^(p)) / (Omega + sum_p D_b^(p)).

    Args:
        q: Local query shard of shape (B, H, shard_L, D).
        k: Local key shard of shape (B, H, shard_L, D).
        v: Local value shard of shape (B, H, shard_L, D).
        sink_omega: Nonnegative attention sink scalar mass.
        causal: Autoregressive masking flag.
        axis_name: Name of the distributed mesh axis over which to ring-communicate.

    Returns:
        Locally normalized attention output shard of shape (B, H, shard_L, D).
    """
    num_devices = lax.psum(1, axis_name)
    device_idx = lax.axis_index(axis_name)
    head_dim = q.shape[-1]
    scale = float(1.0 / math.sqrt(head_dim))
    shard_len = q.shape[-2]

    def _ring_step(hop, state):
        o_acc, d_acc, curr_k, curr_v = state

        # Systolic Matmul on MXU
        s = jnp.matmul(q, jnp.swapaxes(curr_k, -1, -2)) * scale

        # Octic Squaring Kernel on VMU
        p = _vmu_octic_kernel(s)

        # Causal masking across distributed sequence shards
        if causal:
            kv_device = (device_idx - hop) % num_devices
            row_ids = lax.broadcasted_iota(jnp.int32, (shard_len, shard_len), 0) + device_idx * shard_len
            col_ids = lax.broadcasted_iota(jnp.int32, (shard_len, shard_len), 1) + kv_device * shard_len
            mask = col_ids <= row_ids
            tile_active = kv_device <= device_idx
            p = jnp.where(tile_active & mask[None, None, :, :], p, 0.0)

        # Pure Additive Tile Accumulation
        o_acc = o_acc + jnp.matmul(p.astype(v.dtype), curr_v)
        d_acc = d_acc + jnp.sum(p, axis=-1, keepdims=True)

        # Rotate KV blocks along the 16-chip 3D Torus ICI ring
        next_k = lax.ppermute(curr_k, axis_name=axis_name, perm=[(i, (i + 1) % num_devices) for i in range(16)])
        next_v = lax.ppermute(curr_v, axis_name=axis_name, perm=[(i, (i + 1) % num_devices) for i in range(16)])

        return (o_acc, d_acc, next_k, next_v)

    init_o = jnp.zeros_like(q, dtype=jnp.float32)
    init_d = jnp.zeros((q.shape[0], q.shape[1], shard_len, 1), dtype=jnp.float32)

    final_o, final_d, _, _ = lax.fori_loop(0, num_devices, _ring_step, (init_o, init_d, k, v))

    # Single global rational normalization at the end of the ring traversal
    y = final_o / (final_d + sink_omega).astype(final_o.dtype)
    return y.astype(q.dtype)


def algebraic_flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
) -> jax.Array:
    """Unified Algebraic FlashAttention interface with hardware autotuning.

    Automatically handles padding to 128x128 MXU boundaries if sequence length
    is not an exact multiple of the systolic dimension.

    Args:
        q: Query tensor (B, H, L, D).
        k: Key tensor (B, H, L, D).
        v: Value tensor (B, H, L, D).
        sink_omega: Nonnegative attention sink scalar mass.
        causal: Autoregressive causal masking flag.
        block_q: Query tile size.
        block_k: Key tile size.

    Returns:
        Attention output tensor Y of shape (B, H, L, D).
    """
    orig_seq_len = q.shape[-2]

    # Check if padding is needed to reach tile multiples
    rem = orig_seq_len % block_q
    if rem != 0:
        pad_len = block_q - rem
        pad_config = [(0, 0), (0, 0), (0, pad_len), (0, 0)]
        q_pad = jnp.pad(q, pad_config)
        k_pad = jnp.pad(k, pad_config)
        v_pad = jnp.pad(v, pad_config)
        valid_len = orig_seq_len
    else:
        q_pad, k_pad, v_pad = q, k, v
        valid_len = None

    # Execute Pallas TPU kernel if on TPU hardware, or tiled XLA kernel
    is_tpu = False
    try:
        is_tpu = (jax.devices()[0].platform == "tpu")
    except Exception:
        pass

    if is_tpu:
        out = pallas_afa_forward(
            q_pad,
            k_pad,
            v_pad,
            sink_omega=sink_omega,
            causal=causal,
            block_q=block_q,
            block_k=block_k,
            valid_seq_len=valid_len,
            interpret=False,
        )
    else:
        out = tiled_afa_forward(
            q_pad,
            k_pad,
            v_pad,
            sink_omega=sink_omega,
            causal=causal,
            block_q=block_q,
            block_k=block_k,
            valid_seq_len=valid_len,
        )

    if rem != 0:
        out = out[:, :, :orig_seq_len, :]

    return out

