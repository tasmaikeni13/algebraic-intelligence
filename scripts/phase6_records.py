"""Phase 6 evidence fingerprints, including the inherited dependency closure."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.phase5_records import source_hashes as phase5_hashes, environment as phase5_environment

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Preserve NumPy scalar values as JSON scalars; reject NaN and infinity."""
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported evidence value: {type(item).__name__}")
    encoded = json.dumps(value, indent=2, allow_nan=False, default=scalar) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def source_hashes():
    result = phase5_hashes()
    names = [
        'src/kernels/__init__.py',
        'src/kernels/pallas_afa.py',
        'tests/reference_attention.py',
        'tests/test_pallas_afa.py',
        'formal/AlgebraicTheory/Kernel.lean',
        'formal/AlgebraicTheory/Gate.lean',
        'phases/phase6.md',
    ]
    names += [
        str(p.relative_to(ROOT))
        for pattern in ('phase6_*.py', '*phase6*.py', 'run_verify_pallas.py', 'audit_xla_hlo.py')
        for p in (ROOT / 'scripts').glob(pattern)
    ]
    for name in sorted(set(names)):
        p = ROOT / name
        if p.exists():
            result[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def environment():
    env = phase5_environment()
    env['source_sha256'] = source_hashes()
    return env
