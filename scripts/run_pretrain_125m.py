#!/usr/bin/env python3
"""Run one Phase 9 125M/2.5B-token pretraining arm on a TPU v4-32 slice."""

import argparse
import os
from pathlib import Path
import sys

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "tpu,cpu"
os.environ.setdefault("JAX_ENABLE_X64", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
from jax.experimental import multihost_utils

from scripts.phase8_records import write_json
from scripts.phase9_experiments import (
    PHASE9_SEEDS,
    PHASE9_TOKEN_BUDGET,
    latest_checkpoint,
    load_verified_phase8_configs,
)
from scripts.phase9_records import environment, source_hashes
from scripts.run_hparam_sweep import audit_phase8_ast, run_sweep_arm
from src.mesh import create_tpu_mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("algebraic", "standard"), required=True)
    parser.add_argument("--seed", type=int, choices=PHASE9_SEEDS, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--phase8-results", type=Path, default=ROOT / "results/phase8")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-tokens", type=int, default=PHASE9_TOKEN_BUDGET)
    parser.add_argument("--checkpoint-every-tokens", type=int, default=100_000_000)
    parser.add_argument("--max-checkpoints", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if args.total_tokens < PHASE9_TOKEN_BUDGET:
        raise ValueError("Phase 9 evidence requires at least 2,500,000,000 tokens per run")

    internal_architecture = "baseline" if args.architecture == "standard" else "algebraic"
    hparams = load_verified_phase8_configs(args.phase8_results)[internal_architecture]
    train_path = args.data_dir / "fineweb_train_2_5B.npy"
    valid_path = args.data_dir / "fineweb_valid.npy"
    if not (train_path.exists() and valid_path.exists()):
        raise FileNotFoundError(
            "Phase 9 requires data/fineweb_train_2_5B.npy and data/fineweb_valid.npy"
        )

    is_multi_host = False
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        jax.distributed.initialize(initialization_timeout=120)
        is_multi_host = True

    devices = jax.devices()
    platform = devices[0].platform
    process_index = jax.process_index()
    process_count = jax.process_count()
    if platform != "tpu" or len(devices) != 16 or process_count != 4:
        raise RuntimeError(
            "Phase 9 requires four processes and 16 TPU devices; "
            f"found {process_count} processes and {len(devices)} {platform} devices"
        )
    ast_audit = audit_phase8_ast()
    if not ast_audit["passed"]:
        raise RuntimeError(f"algebraic source audit failed: {ast_audit['violations']}")

    output = args.output_dir.resolve()
    checkpoints = output / "checkpoints"
    resume = args.resume
    if resume is None:
        resume = latest_checkpoint(checkpoints)
    mesh = create_tpu_mesh(devices=devices)

    run = run_sweep_arm(
        architecture=internal_architecture,
        candidate_name="phase8-selected",
        seed=args.seed,
        hparams=hparams,
        token_path=train_path,
        valid_path=valid_path,
        total_tokens=args.total_tokens,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        mesh=mesh,
        log_every=args.log_every,
        is_multi_host=is_multi_host,
        checkpoint_dir=checkpoints,
        checkpoint_every_tokens=args.checkpoint_every_tokens,
        resume_checkpoint=resume,
        checkpoint_metadata={"source_sha256": source_hashes()},
        max_checkpoints=args.max_checkpoints,
    )

    passed = (
        run["total_tokens"] >= PHASE9_TOKEN_BUDGET
        and run["nan_or_inf_count"] == 0
        and run["loss_spike_count"] == 0
        and run["peak_gradient_norm"] <= 5.0
        and (
            internal_architecture != "algebraic"
            or (
                run["normalization_second_moment_min"] >= 0.8
                and run["normalization_second_moment_max"] <= 1.3
            )
        )
    )
    record = {
        "phase": 9,
        "gate_version": 1,
        "scope": "single_pretraining_arm",
        "environment": environment(),
        "hardware": {
            "platform": platform,
            "device_count": len(devices),
            "process_count": process_count,
            "mesh_shape": dict(mesh.shape),
        },
        "ast_audit": ast_audit,
        "run": run,
        "passed": passed,
    }
    if source_hashes() != record["environment"]["source_sha256"]:
        record["passed"] = False
        record["error"] = "Source changed during execution"
    if process_index == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "run_metrics.json", record)

    if is_multi_host:
        multihost_utils.sync_global_devices(
            f"phase9-{args.architecture}-{args.seed}-complete"
        )
        jax.distributed.shutdown()
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
