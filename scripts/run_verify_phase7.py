#!/usr/bin/env python3
"""Verification entrypoint for Phase 7 Full Architecture Assembly & Pilot Pretraining."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import time
from pathlib import Path
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.phase7_records import hardware_evidence, source_hashes, write_json, environment
from scripts.phase7_experiments import audit_phase7_ast
from scripts.phase6_records import source_hashes as phase6_source_hashes


def main():
    print("================================================================================")
    print("PHASE 7 VERIFICATION: FULL ARCHITECTURE ASSEMBLY & PILOT PRETRAINING")
    print("================================================================================")
    t0 = time.time()

    # 1. Inherited Phase 6 Check
    phase6_path = ROOT / "results/phase6/metrics.json"
    phase6 = json.loads(phase6_path.read_text()) if phase6_path.exists() else {}
    inherited_phase6 = {
        "metrics_exist": phase6_path.exists(),
        "pass_record_exists": (ROOT / "results/phase6/PASS.md").exists(),
        "status_pass": phase6.get("status") == "PASS" and phase6.get("passed") is True,
        "source_matches": phase6.get("environment", {}).get("source_sha256") == phase6_source_hashes(),
    }
    inherited_phase6["passed"] = all(inherited_phase6.values())

    # 2. Formal Lean 4 Verification
    lake = shutil.which("lake") or str(Path.home() / ".elan/bin/lake")
    build = subprocess.run([lake, "build"], cwd=ROOT / "formal", capture_output=True, text=True)
    res_dir = ROOT / "results/phase7"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "lean-build.log").write_text(build.stdout + build.stderr)

    proof_violations = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "formal").rglob("*.lean")
        if ".lake" not in path.parts and re.search(r"\b(sorry|admit|axiom)\b", path.read_text())
    ]
    formal_metrics = {
        "passed": build.returncode == 0
        and not re.search("warning", build.stdout + build.stderr, re.I)
        and not proof_violations,
        "proof_scan_violations": proof_violations,
    }

    # 3. AST Zero-Transcendental Audit
    ast_audit = audit_phase7_ast()

    # 4. Unit and Integration Tests
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    (res_dir / "pytest.log").write_text(tests.stdout + tests.stderr)
    unit_tests = {"passed": tests.returncode == 0}

    # 5. Hardware TPU Pilot Evidence
    tpu_metrics_path = res_dir / "tpu/metrics.json"
    hardware = hardware_evidence(tpu_metrics_path, source_hashes())

    metrics = {
        "phase": 7,
        "gate_version": 1,
        "environment": environment(),
        "inherited_phase6": inherited_phase6,
        "formal": formal_metrics,
        "ast_audit": ast_audit,
        "unit_tests": unit_tests,
        "hardware": hardware,
        "passed": (
            inherited_phase6["passed"]
            and formal_metrics["passed"]
            and ast_audit["passed"]
            and unit_tests["passed"]
            and hardware["passed"]
        ),
    }

    if source_hashes() != metrics["environment"]["source_sha256"]:
        metrics["source_changed"] = True
        metrics["passed"] = False

    metrics["status"] = "PASS" if metrics["passed"] else "FAIL"
    metrics["elapsed_seconds"] = time.time() - t0
    output_path = res_dir / "metrics.json"
    write_json(output_path, metrics)

    print("\n================================================================================")
    print(f"PHASE 7 AGGREGATE VERIFICATION COMPLETE: {metrics['status']}")
    print(f"Total elapsed: {metrics['elapsed_seconds']:.2f}s")
    print(f"Metrics written to: {output_path}")
    print("================================================================================")

    if not metrics["passed"]:
        print("Details:")
        for k in ("inherited_phase6", "formal", "ast_audit", "unit_tests", "hardware"):
            print(f"  {k:20s}: {metrics[k].get('passed', False)}")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
