"""Unit and integration tests for Phase 7 models (AlgebraicTransformerLM & StandardTransformerLM).

Verifies:
1. Shape and parameter-budget contracts (~15M-20M matched scale).
2. Numerical stability: forward pass finite, loss finite, backward gradients finite.
3. Zero-transcendental AST audit of src/model.py.
4. Mesh creation and sharding specifications in src/mesh.py.
"""

import ast
from pathlib import Path
import pytest

import jax
import jax.numpy as jnp
import numpy as np

from src.model import AlgebraicTransformerLM, ModelConfig, count_parameters
from src.baseline import StandardTransformerLM, BaselineConfig
from src.mesh import create_tpu_mesh, ModelSharding
from scripts.audit_primitives import source_audit, FORBIDDEN


def test_parameter_count_parity():
    """Verify both architectures have matched parameter count within <= 1%."""
    # Small test config to verify parity at arbitrary scales
    cfg_alg = ModelConfig(vocab_size=1000, d_model=64, num_layers=2, num_heads=2, d_ff=128, max_seq_len=64)
    cfg_base = BaselineConfig(vocab_size=1000, d_model=64, num_layers=2, num_heads=2, d_ff=128, max_seq_len=64)

    alg_model = AlgebraicTransformerLM(cfg_alg)
    base_model = StandardTransformerLM(cfg_base)

    key = jax.random.PRNGKey(42)
    params_alg = alg_model.init_params(key)
    params_base = base_model.init_params(key)

    n_alg = count_parameters(params_alg)
    n_base = count_parameters(params_base)

    # Base has learnable gamma parameters in RMSNorm; AVN is parameter-free.
    # Exclude AVN parameter savings per Phase 7 specification, parity is within 1%.
    diff_pct = abs(n_alg - n_base) / max(n_alg, n_base)
    assert diff_pct <= 0.01, f"Parameter mismatch: alg={n_alg}, base={n_base}, delta={diff_pct:.4f}"


def test_algebraic_model_forward_and_loss():
    """Verify forward and backward passes execute cleanly with finite outputs."""
    cfg = ModelConfig(vocab_size=256, d_model=48, num_layers=2, num_heads=2, d_ff=96, max_seq_len=32, dtype=jnp.float32)
    model = AlgebraicTransformerLM(cfg)

    key = jax.random.PRNGKey(123)
    params = model.init_params(key)

    B, T = 2, 16
    tokens = jax.random.randint(key, (B, T), 0, cfg.vocab_size)
    targets = jax.random.randint(jax.random.fold_in(key, 1), (B, T), 0, cfg.vocab_size)

    # Forward pass
    logits = model.forward(params, tokens)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert jnp.all(jnp.isfinite(logits))

    # Loss and gradient
    def loss_fn(p):
        loss_val, _ = model.loss(p, tokens, targets)
        return loss_val

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert jnp.isfinite(loss)
    assert loss > 0.0

    # Verify all gradients are finite
    grad_leaves = jax.tree_util.tree_leaves(grads)
    for g in grad_leaves:
        assert jnp.all(jnp.isfinite(g)), "Non-finite gradient encountered"


def test_standard_model_forward_and_loss():
    """Verify standard baseline forward and backward passes execute cleanly."""
    cfg = BaselineConfig(vocab_size=256, d_model=48, num_layers=2, num_heads=2, d_ff=96, max_seq_len=32, dtype=jnp.float32)
    model = StandardTransformerLM(cfg)

    key = jax.random.PRNGKey(456)
    params = model.init_params(key)

    B, T = 2, 16
    tokens = jax.random.randint(key, (B, T), 0, cfg.vocab_size)
    targets = jax.random.randint(jax.random.fold_in(key, 1), (B, T), 0, cfg.vocab_size)

    logits = model.forward(params, tokens)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert jnp.all(jnp.isfinite(logits))

    def loss_fn(p):
        loss_val, _ = model.loss(p, tokens, targets)
        return loss_val

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert jnp.isfinite(loss)
    assert loss > 0.0

    grad_leaves = jax.tree_util.tree_leaves(grads)
    for g in grad_leaves:
        assert jnp.all(jnp.isfinite(g)), "Non-finite baseline gradient encountered"


def test_algebraic_attention_sink_configuration_is_used():
    """Regression: Phase 8 sink candidates must affect the model computation."""
    common = dict(
        vocab_size=64,
        d_model=32,
        num_layers=1,
        num_heads=2,
        d_ff=64,
        max_seq_len=16,
        dtype=jnp.float32,
    )
    low_sink = AlgebraicTransformerLM(ModelConfig(**common, sink_omega=0.1))
    high_sink = AlgebraicTransformerLM(ModelConfig(**common, sink_omega=2.0))
    key = jax.random.PRNGKey(457)
    params = low_sink.init_params(key)
    tokens = jax.random.randint(jax.random.fold_in(key, 1), (1, 12), 0, 64)
    low_logits = low_sink.forward(params, tokens)
    high_logits = high_sink.forward(params, tokens)
    assert not np.allclose(np.asarray(low_logits), np.asarray(high_logits))


def test_zero_transcendental_ast_audit_model():
    """Verify src/model.py contains exactly 0 calls to transcendental functions."""
    path = Path(__file__).resolve().parents[1] / "src/model.py"
    source = path.read_text()
    tree = ast.parse(source)

    violations = source_audit(source)
    assert len(violations) == 0, f"Transcendental violations in src/model.py: {violations}"


def test_mesh_creation():
    """Verify mesh.py creates a valid JAX mesh with ('data', 'fsdp', 'model') axes."""
    mesh = create_tpu_mesh()
    assert "data" in mesh.axis_names
    assert "fsdp" in mesh.axis_names
    assert "model" in mesh.axis_names

    sharding = ModelSharding(mesh)
    assert sharding.replicated is not None
    assert sharding.data_parallel is not None
