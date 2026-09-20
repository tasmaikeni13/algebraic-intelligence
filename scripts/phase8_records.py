"""Phase 8 evidence fingerprints, including the inherited dependency closure."""
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.phase7_records import source_hashes as phase7_hashes, environment as phase7_environment

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Preserve NumPy scalar values as JSON scalars; reject NaN and infinity."""
    def scalar(item):
        if isinstance(item, (np.generic, np.ndarray)):
            if isinstance(item, np.ndarray) and item.ndim == 0:
                return item.item()
            elif isinstance(item, np.ndarray):
                return item.tolist()
            return item.item()
        raise TypeError(f"Unsupported evidence value: {type(item).__name__}")

    encoded = json.dumps(value, indent=2, allow_nan=False, default=scalar) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def source_hashes():
    result = phase7_hashes()
    names = [
        'src/model.py',
        'src/baseline.py',
        'src/mesh.py',
        'src/dataset.py',
        'src/optimizer.py',
        'formal/AlgebraicTheory/Curvature.lean',
        'phases/phase8.md',
        'tests/test_hparam_contracts.py',
    ]
    names += [
        str(p.relative_to(ROOT))
        for pattern in ('phase8_*.py', '*phase8*.py', 'run_hparam_sweep.py', 'prepare_fineweb_edu.py')
        for p in (ROOT / 'scripts').glob(pattern)
    ]
    for name in sorted(set(names)):
        p = ROOT / name
        if p.exists():
            result[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def environment():
    env = phase7_environment()
    env['source_sha256'] = source_hashes()
    return env


def hardware_evidence(path, expected_hashes):
    """Validate that Phase 8 evidence exercised the real hyperparameter sweep on TPU."""
    path = Path(path)
    if not path.exists():
        return {"passed": False, "reason": "Missing TPU evidence"}
    record = json.loads(path.read_text())
    matched = record.get("environment", {}).get("source_sha256") == expected_hashes

    sweep = record.get("sweep_summary", {})
    completed_runs = sweep.get("completed_runs", 0)
    ppl_ratio = sweep.get("mean_perplexity_ratio", 999.0)
    nan_count = sweep.get("nan_or_inf_count", 999)
    spike_count = sweep.get("loss_spike_count", 999)
    peak_grad_norm = sweep.get("peak_gradient_norm", 999.0)
    alg_std_pct = sweep.get("algebraic_seed_std_pct", 999.0)
    base_std_pct = sweep.get("baseline_seed_std_pct", 999.0)

    passed = (
        matched
        and completed_runs == 6
        and ppl_ratio <= 1.08
        and nan_count == 0
        and spike_count == 0
        and peak_grad_norm <= 5.0
        and alg_std_pct < 2.0
        and base_std_pct < 2.0
        and record.get("ast_audit", {}).get("passed", False)
    )

    return {
        "passed": passed,
        "matched_hashes": matched,
        "completed_runs": completed_runs,
        "perplexity_ratio": ppl_ratio,
        "nan_count": nan_count,
        "spike_count": spike_count,
        "peak_gradient_norm": peak_grad_norm,
        "algebraic_seed_std_pct": alg_std_pct,
        "baseline_seed_std_pct": base_std_pct,
    }
