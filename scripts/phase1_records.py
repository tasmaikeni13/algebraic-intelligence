"""Reproducible source fingerprints and artifact serialization."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def source_hashes():
    paths = []
    for folder, pattern in [("src","*.py"),("scripts","*.py"),("tests","*.py"),("formal","*.lean")]:
        paths.extend(p for p in (ROOT/folder).rglob(pattern) if ".lake" not in p.parts)
    paths += [ROOT/p for p in ["requirements.txt","requirements-tpu.txt","formal/lean-toolchain","formal/lake-manifest.json","formal/lakefile.toml","phases/phase1.md"]]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def environment():
    import jax, jaxlib, numpy, scipy
    def git(*args):
        return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
    return {"python":sys.version,"platform":platform.platform(),"hostname":platform.node(),
            "jax":jax.__version__,"jaxlib":jaxlib.__version__,"numpy":numpy.__version__,"scipy":scipy.__version__,
            "git_commit":git("rev-parse","HEAD"),"git_dirty":bool(git("status","--porcelain")),
            "source_sha256":source_hashes()}
