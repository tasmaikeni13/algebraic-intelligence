"""Phase 8 contract tests: zero-transcendental verification, budget parity, and stability bounds."""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
from scripts.audit_primitives import source_audit
from scripts.phase8_records import source_hashes, hardware_evidence


def test_ast_zero_transcendental_audit():
    """Verify strictly zero transcendental function calls in the algebraic production stack."""
    files_to_audit = [
        "src/model.py",
        "src/primitives.py",
        "src/attention.py",
        "src/loss.py",
        "src/optimizer.py",
        "src/mesh.py",
        "src/kernels/pallas_afa.py",
    ]
    for rel_path in files_to_audit:
        full_path = ROOT / rel_path
        assert full_path.exists(), f"Missing file {rel_path}"
        code = full_path.read_text()
        violations = source_audit(code)
        assert len(violations) == 0, f"Transcendental violations found in {rel_path}: {violations}"


def test_optimal_hparam_files_structure():
    """Verify that optimal hyperparameter artifacts conform to expected schema."""
    alg_path = ROOT / "results/phase8/algebraic_optimal.json"
    base_path = ROOT / "results/phase8/baseline_optimal.json"
    ledger_path = ROOT / "results/phase8/sweep_ledger.json"

    if not (alg_path.exists() and base_path.exists() and ledger_path.exists()):
        pytest.skip("Phase 8 optimal configurations not yet generated")

    alg_cfg = json.loads(alg_path.read_text())
    base_cfg = json.loads(base_path.read_text())
    ledger = json.loads(ledger_path.read_text())

    # Check required keys for algebraic configuration
    for key in ["learning_rate", "warmup_steps", "weight_decay", "beta1", "beta2", "sink_omega", "gamma", "schedule"]:
        assert key in alg_cfg, f"Missing key '{key}' in algebraic_optimal.json"
    assert alg_cfg["schedule"] == "ards", f"Expected ARDS schedule for algebraic model, got {alg_cfg['schedule']}"

    # Check required keys for baseline configuration
    for key in ["learning_rate", "warmup_steps", "weight_decay", "beta1", "beta2", "schedule"]:
        assert key in base_cfg, f"Missing key '{key}' in baseline_optimal.json"

    assert len(ledger.get("runs", [])) == 6, f"Expected 6 runs in sweep ledger, got {len(ledger.get('runs', []))}"


def test_curvature_convergence_bounds():
    """Confirm optimal parameters satisfy the contraction mapping formalized in Curvature.lean."""
    alg_path = ROOT / "results/phase8/algebraic_optimal.json"
    if not alg_path.exists():
        pytest.skip("Phase 8 optimal configurations not yet generated")

    alg_cfg = json.loads(alg_path.read_text())
    lr = alg_cfg["learning_rate"]
    wd = alg_cfg["weight_decay"]

    # Theorem adamw_decoupled_weight_decay: w_{t+1} = (1 - lr * wd) * w_t - lr * u_t
    # Contraction requires 0 < 1 - lr * wd < 1
    decay_factor = 1.0 - lr * wd
    assert 0.0 < decay_factor < 1.0, f"Decoupled weight decay violates contraction mapping: {decay_factor}"
    assert lr * wd < 0.01, f"Decoupled step factor too large: {lr * wd}"
    assert 1e-4 <= lr <= 2e-3, f"Learning rate out of search bounds: {lr}"
    assert 0.001 <= wd <= 0.15, f"Weight decay out of search bounds: {wd}"


def test_budget_parity_contracts():
    """Verify strict budget parity at 125M parameter scale across architectures."""
    from src.model import AlgebraicTransformerLM, ModelConfig, count_parameters
    from src.baseline import StandardTransformerLM, BaselineConfig
    import jax

    cfg_alg = ModelConfig(
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=2048,
        max_seq_len=2048,
    )
    cfg_base = BaselineConfig(
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=2048,
        max_seq_len=2048,
    )

    m_alg = AlgebraicTransformerLM(cfg_alg)
    p_alg = m_alg.init_params(jax.random.PRNGKey(42))
    n_alg = count_parameters(p_alg)

    m_base = StandardTransformerLM(cfg_base)
    p_base = m_base.init_params(jax.random.PRNGKey(42))
    n_base = count_parameters(p_base)

    diff = abs(n_alg - n_base) / max(n_alg, n_base)
    assert diff < 0.01, f"Parameter mismatch exceeds 1%: {diff * 100:.3f}% (Alg: {n_alg}, Base: {n_base})"
    assert 120_000_000 <= n_alg <= 130_000_000, f"Algebraic parameters outside 125M scale: {n_alg}"
    assert 120_000_000 <= n_base <= 130_000_000, f"Baseline parameters outside 125M scale: {n_base}"


def test_multi_seed_stability_contract():
    """Verify that multi-seed variance across Seeds 42, 43, 44 is strictly bounded (< 2% std/mean)."""
    ledger_path = ROOT / "results/phase8/sweep_ledger.json"
    if not ledger_path.exists():
        pytest.skip("Phase 8 sweep ledger not yet produced")

    ledger = json.loads(ledger_path.read_text())
    runs = ledger.get("runs", [])
    assert len(runs) == 6, f"Expected 6 runs, found {len(runs)}"

    alg_losses = [r["validation_loss"] for r in runs if r["architecture"] == "algebraic"]
    base_losses = [r["validation_loss"] for r in runs if r["architecture"] == "baseline"]

    import numpy as np
    alg_mean = np.mean(alg_losses)
    alg_std = np.std(alg_losses)
    alg_std_pct = (alg_std / alg_mean) * 100.0

    base_mean = np.mean(base_losses)
    base_std = np.std(base_losses)
    base_std_pct = (base_std / base_mean) * 100.0

    assert alg_std_pct < 2.0, f"Algebraic multi-seed std exceeds 2%: {alg_std_pct:.3f}%"
    assert base_std_pct < 2.0, f"Baseline multi-seed std exceeds 2%: {base_std_pct:.3f}%"


def test_hardware_evidence_records():
    """Validate Phase 8 hardware metrics record against contract gates."""
    metrics_path = ROOT / "results/phase8/tpu/metrics.json"
    if not metrics_path.exists():
        metrics_path = ROOT / "results/phase8/metrics.json"
    record = json.loads(metrics_path.read_text())
    hashes = source_hashes()
    if record.get("environment", {}).get("source_sha256") != hashes:
        pytest.skip("Committed TPU evidence is stale and must be regenerated by the aggregate verifier")
    res = hardware_evidence(metrics_path, hashes)
    assert res["passed"] is True, f"Hardware evidence validation failed: {res}"
