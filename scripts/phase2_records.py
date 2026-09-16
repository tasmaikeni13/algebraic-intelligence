"""Phase 2 evidence fingerprints, including the inherited dependency closure."""
import hashlib
from pathlib import Path
from scripts.phase1_records import source_hashes as phase1_hashes, environment as phase1_environment, write_json

ROOT=Path(__file__).resolve().parents[1]


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
