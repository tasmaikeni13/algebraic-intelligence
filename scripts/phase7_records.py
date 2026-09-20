"""Phase 7 evidence fingerprints, including the inherited dependency closure."""
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.phase6_records import source_hashes as phase6_hashes, environment as phase6_environment

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
    result = phase6_hashes()
    names = [
        'src/model.py',
        'src/baseline.py',
        'src/mesh.py',
        'src/dataset.py',
        'tests/test_model.py',
        'formal/AlgebraicTheory/Composition.lean',
        'phases/phase7.md',
    ]
    names += [
        str(p.relative_to(ROOT))
        for pattern in ('phase7_*.py', '*phase7*.py', 'run_pilot_15m.py')
        for p in (ROOT / 'scripts').glob(pattern)
    ]
    for name in sorted(set(names)):
        p = ROOT / name
        if p.exists():
            result[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def environment():
    env = phase6_environment()
    env['source_sha256'] = source_hashes()
    return env


def hardware_evidence(path, expected_hashes):
    """Validate that Phase 7 evidence exercised the real pilot pretraining on TPU."""
    path = Path(path)
    if not path.exists():
        return {"passed": False, "reason": "Missing TPU evidence"}
    record = json.loads(path.read_text())
    matched = record.get("environment", {}).get("source_sha256") == expected_hashes

    pilot = record.get("pilot_pretraining", {})
    ppl_ratio = pilot.get("perplexity_ratio", 999.0)
    nan_count = pilot.get("nan_or_inf_count", 999)
    spike_count = pilot.get("loss_spike_count", 999)
    peak_grad_norm = pilot.get("peak_gradient_norm", 999.0)
    throughput_ratio = pilot.get("throughput_ratio", 0.0)

    passed = (
        matched
        and ppl_ratio <= 1.08
        and nan_count == 0
        and spike_count == 0
        and peak_grad_norm <= 5.0
        and throughput_ratio >= 0.90
        and record.get("ast_audit", {}).get("passed", False)
    )

    return {
        "passed": passed,
        "matched_hashes": matched,
        "perplexity_ratio": ppl_ratio,
        "nan_count": nan_count,
        "spike_count": spike_count,
        "peak_gradient_norm": peak_grad_norm,
        "throughput_ratio": throughput_ratio,
    }
