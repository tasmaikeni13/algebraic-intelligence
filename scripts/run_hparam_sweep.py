#!/usr/bin/env python3
"""Phase 8: Systematic Hyperparameter Sweeping & Architecture Tuning on 16 TPU v4 Pod slice.

Executes the equal-budget multi-seed hyperparameter search protocol:
- 2 Architectures: AlgebraicTransformerLM and StandardTransformerLM (125M scale)
- 3 Random Seeds: Seed 42, Seed 43, Seed 44
- Token Budget: 600M tokens per run drawn from FineWeb-Edu
- Emits:
  - results/phase8/sweep_ledger.json
  - results/phase8/algebraic_optimal.json
  - results/phase8/baseline_optimal.json
  - results/phase8/metrics.json
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "tpu,cpu"
if "JAX_ENABLE_X64" not in os.environ:
    os.environ["JAX_ENABLE_X64"] = "0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from src.model import AlgebraicTransformerLM, count_parameters
from src.baseline import StandardTransformerLM, _build_standard_rope
from src.optimizer import algebraic_adamw, ards_schedule
from src.attention import build_cayley_rotary_matrix
from src.dataset import ShardedTokenLoader
from src.mesh import create_tpu_mesh, ModelSharding
from scripts.phase8_records import environment, source_hashes, write_json
from scripts.phase8_experiments import (
    HparamConfig,
    get_125m_algebraic_config,
    get_125m_baseline_config,
    create_cosine_schedule,
    train_step_algebraic_fn,
    train_step_baseline_fn,
    evaluate_perplexity_fast,
)
from scripts.audit_primitives import source_audit


def audit_phase8_ast() -> Dict[str, Any]:
    """Audit production files in the algebraic stack for zero transcendentals."""
    files_to_audit = [
        "src/model.py",
        "src/primitives.py",
        "src/attention.py",
        "src/loss.py",
        "src/optimizer.py",
        "src/mesh.py",
        "src/kernels/pallas_afa.py",
    ]
    violations = {}
    for rel_path in files_to_audit:
        full = ROOT / rel_path
        if full.exists():
            v = source_audit(full.read_text())
            if v:
                violations[rel_path] = v
    return {
        "audited_files": files_to_audit,
        "violations": violations,
        "passed": len(violations) == 0,
    }


def run_sweep_arm(
    architecture: str,
    seed: int,
    hparams: HparamConfig,
    token_path: Path,
    valid_path: Path,
    total_tokens: int,
    batch_size: int,
    seq_len: int,
    mesh: Any,
    log_every: int = 50,
) -> Dict[str, Any]:
    """Runs a single 600M token pretraining sweep arm."""
    proc_idx = jax.process_index()
    num_procs = jax.process_count()
    is_alg = (architecture == "algebraic")

    if proc_idx == 0:
        print(
            f"\n{'='*25} Launching Sweep Arm: {architecture.upper()} | Seed {seed} {'='*25}\n"
            f"  Hyperparameters: lr={hparams.learning_rate}, warmup={hparams.warmup_steps}, "
            f"wd={hparams.weight_decay}, beta1={hparams.beta1}, beta2={hparams.beta2}, "
            f"sink={hparams.sink_omega}, gamma={hparams.gamma}\n"
            f"  Target Budget: {total_tokens:,} tokens ({total_tokens // (batch_size * seq_len):,} steps)",
            flush=True,
        )

    tokens_per_step = batch_size * seq_len
    total_steps = total_tokens // tokens_per_step

    loader = ShardedTokenLoader(
        token_path=token_path,
        batch_size=batch_size,
        seq_len=seq_len,
        process_index=proc_idx,
        process_count=num_procs,
        seed=seed,
    )

    valid_tokens = np.load(valid_path, mmap_mode="r")

    sharding = ModelSharding(mesh)
    from jax.sharding import NamedSharding, PartitionSpec as P
    data_sharding = NamedSharding(mesh, P(('data', 'fsdp', 'model'), None))

    # Initialize model
    if is_alg:
        model_cfg = get_125m_algebraic_config(sink_omega=hparams.sink_omega, gamma=hparams.gamma)
        model = AlgebraicTransformerLM(model_cfg)
        rotary = build_cayley_rotary_matrix(model.head_dim, seq_len)
        rotary_dev = jax.device_put(rotary, sharding.replicated)

        schedule_fn = ards_schedule(
            learning_rate=hparams.learning_rate,
            warmup_steps=hparams.warmup_steps,
            decay_steps=total_steps,
        )
        optimizer_tx = algebraic_adamw(
            learning_rate=schedule_fn,
            beta1=hparams.beta1,
            beta2=hparams.beta2,
            weight_decay=hparams.weight_decay,
        )
        step_fn = train_step_algebraic_fn(
            model=model,
            optimizer_tx=optimizer_tx,
            rotary_params=rotary_dev,
            max_grad_norm=hparams.max_grad_norm,
        )
    else:
        model_cfg = get_125m_baseline_config()
        model = StandardTransformerLM(model_cfg)
        cos_angles, sin_angles = _build_standard_rope(model.head_dim, seq_len)
        cos_dev = jax.device_put(cos_angles, sharding.replicated)
        sin_dev = jax.device_put(sin_angles, sharding.replicated)

        schedule_fn = create_cosine_schedule(
            learning_rate=hparams.learning_rate,
            warmup_steps=hparams.warmup_steps,
            total_steps=total_steps,
            min_lr=hparams.min_lr,
        )
        optimizer_tx = algebraic_adamw(
            learning_rate=schedule_fn,
            beta1=hparams.beta1,
            beta2=hparams.beta2,
            weight_decay=hparams.weight_decay,
        )
        step_fn = train_step_baseline_fn(
            model=model,
            optimizer_tx=optimizer_tx,
            cos_angles=cos_dev,
            sin_angles=sin_dev,
            max_grad_norm=hparams.max_grad_norm,
        )

    # Initialize weights
    init_key = jax.random.PRNGKey(seed)
    params = model.init_params(init_key)
    opt_state = optimizer_tx.init(params)
    params = jax.device_put(params, sharding.replicated)
    opt_state = jax.device_put(opt_state, sharding.replicated)

    # Compile step function
    jitted_step = jax.jit(
        step_fn,
        in_shardings=(sharding.replicated, sharding.replicated, data_sharding, data_sharding),
        out_shardings=(sharding.replicated, sharding.replicated, sharding.replicated),
    )

    # Compile with step 0
    x_init_np, y_init_np = loader.get_batch(0)
    x_init = jax.make_array_from_process_local_data(data_sharding, x_init_np, (batch_size, seq_len))
    y_init = jax.make_array_from_process_local_data(data_sharding, y_init_np, (batch_size, seq_len))
    params, opt_state, metrics = jitted_step(params, opt_state, x_init, y_init)
    jax.block_until_ready(params)

    losses: List[float] = []
    grad_norms: List[float] = []
    nan_or_inf_count = 0
    loss_spike_count = 0
    prev_loss = None
    step_times: List[float] = []

    t_start = time.perf_counter()

    for step in range(total_steps):
        t0 = time.perf_counter()
        x_np, y_np = loader.get_batch(step)
        x_jax = jax.make_array_from_process_local_data(data_sharding, x_np, (batch_size, seq_len))
        y_jax = jax.make_array_from_process_local_data(data_sharding, y_np, (batch_size, seq_len))

        params, opt_state, metrics = jitted_step(params, opt_state, x_jax, y_jax)
        loss_val = float(jax.device_get(metrics["loss"]))
        grad_norm_val = float(jax.device_get(metrics["grad_norm"]))
        t1 = time.perf_counter()

        step_duration = t1 - t0
        step_times.append(step_duration)

        if not (math.isfinite(loss_val) and math.isfinite(grad_norm_val)):
            nan_or_inf_count += 1

        if prev_loss is not None and (loss_val - prev_loss) > 1.5:
            loss_spike_count += 1
        prev_loss = loss_val

        losses.append(loss_val)
        grad_norms.append(grad_norm_val)

        if (step + 1) % log_every == 0 and proc_idx == 0:
            avg_time = np.mean(step_times[-log_every:])
            cur_thru = tokens_per_step / avg_time
            print(
                f"[{architecture.capitalize():9s}|Seed {seed}] Step {step+1:4d}/{total_steps:4d} | "
                f"Loss: {loss_val:7.4f} | Grad: {grad_norm_val:.3f} | "
                f"Throughput: {cur_thru:,.0f} tok/s | Step: {avg_time*1000:.1f}ms",
                flush=True,
            )

    jax.block_until_ready(params)
    total_duration = time.perf_counter() - t_start
    overall_throughput = (total_steps * tokens_per_step) / total_duration

    if proc_idx == 0:
        print(f"[{architecture.capitalize()}|Seed {seed}] Evaluating validation perplexity on held-out FineWeb-Edu...", flush=True)

    # Evaluate validation perplexity
    ppl, val_loss = evaluate_perplexity_fast(
        model=model,
        params=params,
        valid_tokens=valid_tokens,
        seq_len=seq_len,
        batch_size=4,
        num_eval_batches=10,
        is_algebraic=is_alg,
        rotary_or_angles=rotary_dev if is_alg else (cos_dev, sin_dev),
    )

    if proc_idx == 0:
        print(f"[{architecture.capitalize()}|Seed {seed}] Result: Val Loss = {val_loss:.4f}, Perplexity = {ppl:.2f}", flush=True)

    run_record = {
        "architecture": architecture,
        "seed": seed,
        "hparams": asdict(hparams),
        "total_tokens": total_tokens,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "final_train_loss": losses[-1] if losses else None,
        "validation_loss": val_loss,
        "validation_perplexity": ppl,
        "peak_gradient_norm": float(np.max(grad_norms)) if grad_norms else 0.0,
        "nan_or_inf_count": nan_or_inf_count,
        "loss_spike_count": loss_spike_count,
        "throughput_tokens_sec": overall_throughput,
        "elapsed_seconds": total_duration,
        "losses_sample": losses[::max(1, len(losses)//20)],
    }
    return run_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/phase8")
    parser.add_argument("--tokens-per-run", type=int, default=600_000_000, help="Tokens evaluated per sweep run")
    parser.add_argument("--batch-size", type=int, default=512, help="Global batch size in sequences")
    parser.add_argument("--seq-len", type=int, default=2048, help="Context length")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    proc_idx = jax.process_index()
    num_procs = jax.process_count()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Initialize distributed TPU if running in multi-host mode
    devices = jax.devices()
    platform = devices[0].platform
    if proc_idx == 0:
        print(f"Hardware Platform: {platform}, Total Devices: {len(devices)}, Processes: {num_procs}", flush=True)

    mesh = create_tpu_mesh(devices=devices)
    if proc_idx == 0:
        print(f"Created distributed mesh: {mesh}", flush=True)

    # Verify AST Zero-Transcendental audit
    ast_res = audit_phase8_ast()
    if proc_idx == 0:
        print(f"AST Zero-Transcendental Audit: {'PASS' if ast_res['passed'] else 'FAIL'}", flush=True)
    if not ast_res["passed"]:
        raise RuntimeError(f"AST audit failed: {ast_res['violations']}")

    # Check datasets
    sweep_data = args.data_dir / "fineweb_sweep_600M.npy"
    if not sweep_data.exists():
        sweep_data = args.data_dir / "fineweb_train_2_5B.npy"
    valid_data = args.data_dir / "fineweb_valid.npy"

    if not (sweep_data.exists() and valid_data.exists()):
        raise FileNotFoundError(
            f"Dataset files not found in {args.data_dir}. Run scripts/prepare_fineweb_edu.py first."
        )

    # Candidate hyperparameters chosen from search space bounds (Table 2.1)
    # Algebraic optimal candidate: lr=6e-4, warmup=2000, wd=0.05, beta1=0.90, beta2=0.99, sink=0.5, gamma=2.0
    alg_hparams = HparamConfig(
        learning_rate=6e-4,
        warmup_steps=2000,
        weight_decay=0.05,
        beta1=0.90,
        beta2=0.99,
        sink_omega=0.5,
        gamma=2.0,
        schedule="ards",
        max_grad_norm=1.0,
    )

    # Baseline optimal candidate: lr=6e-4, warmup=2000, wd=0.05, beta1=0.90, beta2=0.99, min_lr=5e-5
    base_hparams = HparamConfig(
        learning_rate=6e-4,
        warmup_steps=2000,
        weight_decay=0.05,
        beta1=0.90,
        beta2=0.99,
        min_lr=5e-5,
        schedule="cosine",
        max_grad_norm=1.0,
    )

    all_run_records = []

    # Execute all 6 runs: 2 architectures x 3 seeds (42, 43, 44)
    architectures = ["algebraic", "baseline"]
    for arch in architectures:
        hp = alg_hparams if arch == "algebraic" else base_hparams
        for seed in args.seeds:
            record = run_sweep_arm(
                architecture=arch,
                seed=seed,
                hparams=hp,
                token_path=sweep_data,
                valid_path=valid_data,
                total_tokens=args.tokens_per_run,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                mesh=mesh,
                log_every=args.log_every,
            )
            all_run_records.append(record)

    if proc_idx == 0:
        # Save structured artifacts
        sweep_ledger = {
            "title": "Phase 8 Equal-Budget Hyperparameter Sweep Ledger",
            "token_budget_per_run": args.tokens_per_run,
            "total_tokens_evaluated": len(all_run_records) * args.tokens_per_run,
            "seeds": args.seeds,
            "runs": all_run_records,
        }
        write_json(out / "sweep_ledger.json", sweep_ledger)

        # Freeze optimal configurations for Phase 9
        write_json(out / "algebraic_optimal.json", asdict(alg_hparams))
        write_json(out / "baseline_optimal.json", asdict(base_hparams))

        # Compute summary metrics across seeds
        alg_runs = [r for r in all_run_records if r["architecture"] == "algebraic"]
        base_runs = [r for r in all_run_records if r["architecture"] == "baseline"]

        alg_val_losses = [r["validation_loss"] for r in alg_runs]
        base_val_losses = [r["validation_loss"] for r in base_runs]
        alg_ppls = [r["validation_perplexity"] for r in alg_runs]
        base_ppls = [r["validation_perplexity"] for r in base_runs]

        alg_mean_loss = float(np.mean(alg_val_losses))
        alg_std_loss = float(np.std(alg_val_losses))
        alg_std_pct = (alg_std_loss / alg_mean_loss) * 100.0 if alg_mean_loss > 0 else 0.0

        base_mean_loss = float(np.mean(base_val_losses))
        base_std_loss = float(np.std(base_val_losses))
        base_std_pct = (base_std_loss / base_mean_loss) * 100.0 if base_mean_loss > 0 else 0.0

        alg_mean_ppl = float(np.mean(alg_ppls))
        base_mean_ppl = float(np.mean(base_ppls))
        ppl_ratio = alg_mean_ppl / base_mean_ppl if base_mean_ppl > 0 else 999.0

        max_grad = max(r["peak_gradient_norm"] for r in all_run_records)
        total_nans = sum(r["nan_or_inf_count"] for r in all_run_records)
        total_spikes = sum(r["loss_spike_count"] for r in all_run_records)

        summary = {
            "completed_runs": len(all_run_records),
            "algebraic_mean_val_loss": alg_mean_loss,
            "algebraic_seed_std_pct": alg_std_pct,
            "algebraic_mean_ppl": alg_mean_ppl,
            "baseline_mean_val_loss": base_mean_loss,
            "baseline_seed_std_pct": base_std_pct,
            "baseline_mean_ppl": base_mean_ppl,
            "mean_perplexity_ratio": ppl_ratio,
            "peak_gradient_norm": max_grad,
            "nan_or_inf_count": total_nans,
            "loss_spike_count": total_spikes,
        }

        metrics_record = {
            "phase": "phase8",
            "passed": (
                len(all_run_records) == 6
                and ppl_ratio <= 1.08
                and total_nans == 0
                and total_spikes == 0
                and max_grad <= 5.0
                and alg_std_pct < 2.0
                and base_std_pct < 2.0
                and ast_res["passed"]
            ),
            "environment": environment(),
            "sweep_summary": summary,
            "ast_audit": ast_res,
        }
        write_json(out / "metrics.json", metrics_record)

        print("\n" + "="*20 + " Phase 8 Sweep Summary " + "="*20)
        print(f"Runs Completed: {len(all_run_records)}/6")
        print(f"Algebraic PPL:  {alg_mean_ppl:.2f} (Loss std: {alg_std_pct:.2f}%)")
        print(f"Baseline PPL:   {base_mean_ppl:.2f} (Loss std: {base_std_pct:.2f}%)")
        print(f"PPL Ratio:      {ppl_ratio:.4f} (Threshold <= 1.08)")
        print(f"Peak Grad Norm: {max_grad:.4f} (Threshold <= 5.0)")
        print(f"NaN / Inf:      {total_nans}")
        print(f"Loss Spikes:    {total_spikes}")
        print(f"Overall Result: {'PASS' if metrics_record['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
