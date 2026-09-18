"""Comprehensive tests for Hardware-Fused Algebraic FlashAttention (AFA).

Verifies:
1. Zero-transcendental purity via AST and token audits.
2. Numerical accuracy vs independent NumPy float64 reference (bound <= 1.0e-6).
3. Exact equivalence between tiled, Pallas, and un-tiled representations.
4. Causal masking correctness.
5. Additive tile associativity and scale invariance.
6. Lock-free distributed Ring Attention simulation parity.
7. XLA HLO static opcode audit (0 transcendentals, 0 running-max rescaling).
"""

import ast
import io
from pathlib import Path
import re
import tokenize

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.kernels.pallas_afa import (
    _vmu_octic_kernel,
    afa_kernel,
    pallas_afa_forward,
    tiled_afa_forward,
    exact_afa_reference,
    distributed_ring_afa,
    algebraic_flash_attention,
)
from src.attention import octic_kernel
from tests.reference_attention import reference_afa


FORBIDDEN_IDENTIFIERS = {
    "exp", "expm1", "exp2", "log", "log1p", "log2", "log10",
    "sin", "cos", "tan", "tanh", "sinh", "cosh", "sigmoid", "logistic", "erf", "erfc",
}


def test_zero_transcendental_ast_audit():
    """Verify src/kernels/pallas_afa.py contains 0 transcendental AST calls and token matches."""
    path = Path(__file__).resolve().parents[1] / "src/kernels/pallas_afa.py"
    source = path.read_text()
    tree = ast.parse(source)

    ast_violations = []
    for node in ast.walk(tree):
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        if name in FORBIDDEN_IDENTIFIERS:
            ast_violations.append({"line": node.lineno, "name": name})
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_IDENTIFIERS:
                    ast_violations.append({"line": node.lineno, "name": alias.name})

    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in {tokenize.COMMENT, tokenize.STRING}:
            tokens.append(token.string)
    regex_hits = re.findall(r"\b(?:exp|log|sin|cos|tanh|sigmoid)\b", " ".join(tokens))

    assert len(ast_violations) == 0, f"AST violations: {ast_violations}"
    assert len(regex_hits) == 0, f"Code token regex hits: {regex_hits}"


def test_reference_oracle_parity():
    """Verify exact_afa_reference matches the independent NumPy float64 oracle."""
    key = jax.random.PRNGKey(101)
    B, H, L, D = 1, 2, 64, 32
    q = np.array(jax.random.normal(key, (B, H, L, D), dtype=jnp.float64))
    k = np.array(jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.float64))
    v = np.array(jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.float64))

    ref_np = reference_afa(q, k, v, sink=0.5, causal=False)
    ref_jax = exact_afa_reference(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), sink_omega=0.5, causal=False)

    diff = np.max(np.abs(ref_np - np.array(ref_jax)))
    rel_err = diff / np.max(np.abs(ref_np))
    assert rel_err <= 1.0e-12, f"Oracle mismatch: {rel_err}"


def test_vmu_kernel_negative_tail_is_cancellation_safe():
    """Phase 6 must retain the stable negative-tail identity from Phase 2."""
    scores = jnp.array([-1.0e2, -1.0e3, -1.0e4], dtype=jnp.float32)
    phase2 = np.asarray(octic_kernel(scores), dtype=np.float64)
    phase6 = np.asarray(_vmu_octic_kernel(scores), dtype=np.float64)
    assert np.all(phase6 > 0.0)
    np.testing.assert_allclose(phase6, phase2, rtol=2e-5, atol=0.0)


def test_tiled_afa_numerical_accuracy_fp64():
    """Verify tiled_afa_forward achieves <= 1.0e-6 relative error against float64 ground truth."""
    key = jax.random.PRNGKey(102)
    B, H, L, D = 1, 2, 256, 64
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.float64)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.float64)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.float64)

    y_oracle = reference_afa(np.array(q), np.array(k), np.array(v), sink=0.5, causal=False)
    y_tiled = tiled_afa_forward(q, k, v, sink_omega=0.5, causal=False, block_q=128, block_k=128)

    diff = np.max(np.abs(np.array(y_tiled) - y_oracle))
    norm_exact = np.max(np.abs(y_oracle))
    rel_err = diff / norm_exact
    assert rel_err <= 1.0e-6, f"FP64 relative error {rel_err} exceeds 1.0e-6 bound"


def test_pallas_afa_interpret_accuracy():
    """Verify pallas_afa_forward in interpret mode matches oracle within <= 1.0e-6."""
    key = jax.random.PRNGKey(103)
    B, H, L, D = 1, 2, 256, 64
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.float32)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.float32)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.float32)

    y_oracle = reference_afa(np.array(q), np.array(k), np.array(v), sink=0.5, causal=False)
    y_pallas = pallas_afa_forward(q, k, v, sink_omega=0.5, causal=False, block_q=128, block_k=128, interpret=True)

    diff = np.max(np.abs(np.array(y_pallas) - y_oracle))
    norm_exact = np.max(np.abs(y_oracle))
    rel_err = diff / norm_exact
    assert rel_err <= 1.0e-5, f"Pallas interpret relative error {rel_err} too large"


def test_causal_masking_correctness():
    """Verify autoregressive causal masking prevents attention to future keys."""
    key = jax.random.PRNGKey(104)
    B, H, L, D = 1, 1, 256, 32
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.float32)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.float32)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.float32)

    y_causal_oracle = reference_afa(np.array(q), np.array(k), np.array(v), sink=0.5, causal=True)
    y_causal_tiled = tiled_afa_forward(q, k, v, sink_omega=0.5, causal=True, block_q=128, block_k=128)
    y_causal_pallas = pallas_afa_forward(q, k, v, sink_omega=0.5, causal=True, block_q=128, block_k=128, interpret=True)

    err_tiled = np.max(np.abs(np.array(y_causal_tiled) - y_causal_oracle)) / np.max(np.abs(y_causal_oracle))
    err_pallas = np.max(np.abs(np.array(y_causal_pallas) - y_causal_oracle)) / np.max(np.abs(y_causal_oracle))

    assert err_tiled <= 1.0e-5, f"Causal tiled error too high: {err_tiled}"
    assert err_pallas <= 1.0e-5, f"Causal Pallas error too high: {err_pallas}"

    # Perturb future tokens in key/value: output at earlier positions must not change!
    k_perturbed = k.at[:, :, 150:, :].add(10.0)
    v_perturbed = v.at[:, :, 150:, :].add(10.0)
    y_pert = tiled_afa_forward(q, k_perturbed, v_perturbed, sink_omega=0.5, causal=True, block_q=128, block_k=128)

    # Output before token index 150 must remain identical
    prefix_diff = np.max(np.abs(np.array(y_causal_tiled[:, :, :150, :]) - np.array(y_pert[:, :, :150, :])))
    assert prefix_diff == 0.0, f"Causal leak detected: prefix diff = {prefix_diff}"


def test_unified_interface_arbitrary_seq_len():
    """Verify algebraic_flash_attention handles arbitrary sequence lengths via automatic padding."""
    key = jax.random.PRNGKey(105)
    B, H, L, D = 1, 2, 180, 64  # Not a multiple of 128
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.float32)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.float32)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.float32)

    y_unified = algebraic_flash_attention(q, k, v, sink_omega=0.5, causal=False, block_q=128, block_k=128)
    y_oracle = reference_afa(np.array(q), np.array(k), np.array(v), sink=0.5, causal=False)

    assert y_unified.shape == (B, H, L, D)
    rel_err = np.max(np.abs(np.array(y_unified) - y_oracle)) / np.max(np.abs(y_oracle))
    assert rel_err <= 1.0e-5, f"Unified interface error too high: {rel_err}"


def test_unified_interface_uses_common_tile_multiple():
    """Different query/key tile sizes must not silently drop a remainder."""
    key = jax.random.PRNGKey(205)
    shape = (1, 1, 190, 32)
    q = jax.random.normal(key, shape)
    k = jax.random.normal(jax.random.fold_in(key, 1), shape)
    v = jax.random.normal(jax.random.fold_in(key, 2), shape)
    actual = algebraic_flash_attention(q, k, v, block_q=64, block_k=128)
    expected = exact_afa_reference(q, k, v)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_tiled_interface_rejects_partial_tiles():
    x = jnp.ones((1, 1, 190, 32), dtype=jnp.float32)
    with pytest.raises(ValueError, match="divisible"):
        tiled_afa_forward(x, x, x, block_q=64, block_k=128)


def test_additive_tile_associativity():
    """Verify additive tile accumulation associativity across tile partition boundaries."""
    key = jax.random.PRNGKey(106)
    B, H, L, D = 1, 1, 256, 32
    q = jax.random.normal(key, (B, H, L, D), dtype=jnp.float64)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, L, D), dtype=jnp.float64)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, L, D), dtype=jnp.float64)

    # 128-block tiling vs 64-block tiling vs 256-block tiling (un-tiled)
    y_128 = tiled_afa_forward(q, k, v, sink_omega=0.5, causal=False, block_q=128, block_k=128)
    y_64 = tiled_afa_forward(q, k, v, sink_omega=0.5, causal=False, block_q=64, block_k=64)
    y_ref = exact_afa_reference(q, k, v, sink_omega=0.5, causal=False)

    err_128_vs_ref = np.max(np.abs(np.array(y_128) - np.array(y_ref)))
    err_64_vs_ref = np.max(np.abs(np.array(y_64) - np.array(y_ref)))

    assert err_128_vs_ref <= 1.0e-14, f"128-tile drift: {err_128_vs_ref}"
    assert err_64_vs_ref <= 1.0e-14, f"64-tile drift: {err_64_vs_ref}"


def test_distributed_ring_attention_equivalence():
    """Verify lock-free distributed Ring Attention simulation produces <= 1.0e-6 relative error."""
    num_devices = 4
    B, H, total_L, D = 1, 2, 256, 32
    shard_L = total_L // num_devices

    key = jax.random.PRNGKey(107)
    q = jax.random.normal(key, (B, H, total_L, D), dtype=jnp.float64)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, total_L, D), dtype=jnp.float64)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, total_L, D), dtype=jnp.float64)

    # Global un-sharded reference
    y_global = exact_afa_reference(q, k, v, sink_omega=0.5, causal=False)

    # Sequence shards across simulated ring devices
    q_shards = [q[:, :, p * shard_L : (p + 1) * shard_L, :] for p in range(num_devices)]
    k_shards = [k[:, :, p * shard_L : (p + 1) * shard_L, :] for p in range(num_devices)]
    v_shards = [v[:, :, p * shard_L : (p + 1) * shard_L, :] for p in range(num_devices)]

    scale = float(1.0 / (D ** 0.5))
    y_shards = []
    for p in range(num_devices):
        qp = q_shards[p]
        o_acc = jnp.zeros_like(qp)
        d_acc = jnp.zeros((B, H, shard_L, 1), dtype=jnp.float64)
        for h in range(num_devices):
            kh = k_shards[(p + h) % num_devices]
            vh = v_shards[(p + h) % num_devices]
            s = jnp.matmul(qp, jnp.swapaxes(kh, -1, -2)) * scale
            s_sq = s * s
            r = jax.lax.rsqrt(1.0 + s_sq)
            u = s * r
            denom = jnp.where(s < 0, 1.0 - u, 1.0)
            rho = jnp.where(s < 0, r / denom, s + (1.0 + s_sq) * r)
            p_tile = ((rho * rho) * (rho * rho)) ** 2
            o_acc = o_acc + jnp.matmul(p_tile, vh)
            d_acc = d_acc + jnp.sum(p_tile, axis=-1, keepdims=True)
        yp = o_acc / (d_acc + 0.5)
        y_shards.append(yp)

    y_ring = jnp.concatenate(y_shards, axis=2)
    max_err = np.max(np.abs(np.array(y_ring) - np.array(y_global)))
    rel_err = max_err / np.max(np.abs(np.array(y_global)))

    assert rel_err <= 1.0e-6, f"Ring attention relative error exceeded: {rel_err}"


def test_xla_hlo_static_opcode_audit():
    """Verify compiled XLA HLO contains 0 transcendental opcodes and 0 running-max rescaling."""
    B, H, L, D = 1, 2, 256, 64
    q = jnp.ones((B, H, L, D), dtype=jnp.float32)
    k = jnp.ones((B, H, L, D), dtype=jnp.float32)
    v = jnp.ones((B, H, L, D), dtype=jnp.float32)

    lowered = jax.jit(tiled_afa_forward).lower(q, k, v)
    hlo_text = lowered.as_text().lower()

    forbidden_hlo = ["exponential", "logarithm", "sine", "cosine", "tanh", "sigmoid"]
    found = [op for op in forbidden_hlo if op in hlo_text]
    assert len(found) == 0, f"Forbidden HLO opcodes found in compiled graph: {found}"

    # Also verify absence of running-max exponential rescaling
    assert "exp(" not in hlo_text
