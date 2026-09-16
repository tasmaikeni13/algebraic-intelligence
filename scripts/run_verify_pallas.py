#!/usr/bin/env python3
"""Verification entrypoint for Phase 6 Hardware-Fused Kernels & Algebraic FlashAttention."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.phase6_experiments import run_all_experiments
from scripts.phase6_records import write_json


def main():
    print("================================================================================")
    print("PHASE 6 VERIFICATION: HARDWARE-FUSED ALGEBRAIC FLASHATTENTION (AFA)")
    print("================================================================================")
    t0 = time.time()

    metrics = run_all_experiments()
    output_path = ROOT / "results/phase6/metrics.json"
    write_json(output_path, metrics)

    print("\n================================================================================")
    print(f"PHASE 6 LOCAL CPU VERIFICATION COMPLETE: {metrics['status']}")
    print(f"Total elapsed: {metrics['elapsed_seconds']:.2f}s")
    print(f"Metrics written to: {output_path}")
    print("================================================================================")

    if not metrics["passed"]:
        print("ERROR: One or more Phase 6 local verification gates failed.")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
