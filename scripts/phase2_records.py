"""Phase 2 evidence fingerprints, including the inherited dependency closure."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.phase1_records import source_hashes as phase1_hashes, environment as phase1_environment

ROOT=Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Preserve NumPy scalar values as JSON scalars; reject NaN and infinity."""
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported evidence value: {type(item).__name__}")
    encoded=json.dumps(value,indent=2,allow_nan=False,default=scalar)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(encoded)


def source_hashes():
    result=phase1_hashes()
    names=['src/attention.py','tests/reference_attention.py','tests/test_attention.py',
           'formal/AlgebraicTheory/Kernel.lean','phases/phase2.md']
    names += [str(p.relative_to(ROOT)) for pattern in ('phase2_*.py','*phase2*.py','run_verify_attention.py') for p in (ROOT/'scripts').glob(pattern)]
    for name in sorted(set(names)):
        result[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def environment():
    env=phase1_environment();env['source_sha256']=source_hashes();return env
