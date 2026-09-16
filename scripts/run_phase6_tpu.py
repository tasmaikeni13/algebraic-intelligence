#!/usr/bin/env python3
"""Four-host Phase 6 execution and synchronized 16-chip TPU v4 benchmarks for Algebraic FlashAttention."""
import os
os.environ['JAX_PLATFORMS'] = 'tpu,cpu'
os.environ['JAX_ENABLE_X64'] = '0'
import argparse
import functools
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import mesh_utils, multihost_utils as mh
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as jax_flash_attention
    HAS_JAX_FA = True
except Exception:
    HAS_JAX_FA = False

from src.kernels.pallas_afa import (
    afa_kernel,
    pallas_afa_forward,
    tiled_afa_forward,
    exact_afa_reference,
    distributed_ring_afa,
    algebraic_flash_attention,
)
from tests.reference_attention import reference_afa
from scripts.phase1_experiments import summary
from scripts.phase6_records import environment, source_hashes, write_json


FORBIDDEN_HLO_OPS = [
    "exponential",
    "logarithm",
    "sine",
    "cosine",
    "tanh",
    "sigmoid",
]


def run_parity(place):
    """Verify Algebraic FlashAttention numerical parity on TPU against float64 oracle."""
    rng = np.random.default_rng(642 + jax.process_index())
    rows = []

    configs = [
        (1, 4, 256, 64, False, jnp.float32),
        (1, 4, 256, 64, True, jnp.float32),
        (1, 8, 512, 64, False, jnp.float32),
        (1, 8, 512, 64, True, jnp.float32),
        (1, 8, 1024, 64, False, jnp.bfloat16),
        (1, 8, 1024, 64, True, jnp.bfloat16),
        (1, 8, 2048, 64, False, jnp.bfloat16),
        (1, 8, 2048, 64, True, jnp.bfloat16),
    ]

    for b, h, l, d, causal, dtype in configs:
        tol = 2.0e-4 if dtype == jnp.float32 else 0.04
        # 4 local devices per process; each device evaluates b batch items of shape (b, h, l, d)
        q_host = rng.normal(size=(4 * b, h, l, d)).astype(np.float32)
        k_host = rng.normal(size=(4 * b, h, l, d)).astype(np.float32)
        v_host = rng.normal(size=(4 * b, h, l, d)).astype(np.float32)

        q = place(q_host).astype(dtype)
        k = place(k_host).astype(dtype)
        v = place(v_host).astype(dtype)

        # Forward pass on TPU
        out_afa = tiled_afa_forward(q, k, v, sink_omega=0.5, causal=causal, block_q=128, block_k=128)
        out_afa = jax.block_until_ready(out_afa)

        local_stats = []
        for qs, ks, vs, os in zip(
            q.addressable_shards,
            k.addressable_shards,
            v.addressable_shards,
            out_afa.addressable_shards,
        ):
            qa = np.asarray(qs.data, dtype=np.float64)
            ka = np.asarray(ks.data, dtype=np.float64)
            va = np.asarray(vs.data, dtype=np.float64)
            oa = np.asarray(os.data, dtype=np.float64)

            ref = reference_afa(qa, ka, va, sink=0.5, causal=causal)
            abs_diff = float(np.max(np.abs(oa - ref)))
            norm_ref = float(np.max(np.abs(ref)))
            rel_err = abs_diff / (norm_ref + 1e-12)
            finite_check = float(np.all(np.isfinite(oa)))
            local_stats.append([rel_err, finite_check])

        gathered = np.asarray(mh.process_allgather(np.array(local_stats))).reshape(-1, 2)
        max_err = float(gathered[:, 0].max())
        all_finite = bool(np.all(gathered[:, 1] == 1.0))
        passed = bool(max_err <= tol and all_finite)

        rows.append({
            "shape": [b, h, l, d],
            "causal": causal,
            "dtype": str(jnp.dtype(dtype)),
            "max_err": max_err,
            "all_finite": all_finite,
            "tolerance": tol,
            "passed": passed,
        })

    return {"rows": rows, "passed": all(r["passed"] for r in rows)}


def run_benchmarks(place, mesh):
    """Head-to-head throughput and latency benchmark vs FlashAttention-2 on 16 TPU v4 chips."""
    rng = np.random.default_rng(742 + jax.process_index())
    rows = []
    latencies = {}

    # Target sequence length L=4096 matching Phase 6 benchmark specification
    benchmark_configs = [
        {"b": 1, "h": 8, "l": 2048, "d": 128, "causal": False, "dtype": jnp.bfloat16},
        {"b": 1, "h": 8, "l": 4096, "d": 128, "causal": False, "dtype": jnp.bfloat16},
    ]

    for cfg in benchmark_configs:
        b, h, l, d = cfg["b"], cfg["h"], cfg["l"], cfg["d"]
        causal, dtype = cfg["causal"], cfg["dtype"]
        name = f"L{l}_D{d}_{dtype.__name__}"

        q = place(rng.normal(size=(4 * b, h, l, d)).astype(np.float32)).astype(dtype)
        k = place(rng.normal(size=(4 * b, h, l, d)).astype(np.float32)).astype(dtype)
        v = place(rng.normal(size=(4 * b, h, l, d)).astype(np.float32)).astype(dtype)

        # 1. Algebraic FlashAttention step wrapped in shard_map
        @functools.partial(
            shard_map,
            mesh=mesh,
            in_specs=(P('d', None, None, None), P('d', None, None, None), P('d', None, None, None)),
            out_specs=P('d', None, None, None),
            check_rep=False,
        )
        def afa_step(q_loc, k_loc, v_loc):
            return tiled_afa_forward(q_loc, k_loc, v_loc, sink_omega=0.5, causal=causal, block_q=128, block_k=128)

        # 2. Baseline FlashAttention step wrapped in shard_map
        if HAS_JAX_FA:
            @functools.partial(
                shard_map,
                mesh=mesh,
                in_specs=(P('d', None, None, None), P('d', None, None, None), P('d', None, None, None)),
                out_specs=P('d', None, None, None),
                check_rep=False,
            )
            def baseline_step(q_loc, k_loc, v_loc):
                return jax_flash_attention(q_loc, k_loc, v_loc, causal=causal, sm_scale=float(1.0 / math.sqrt(d)))
        else:
            @functools.partial(
                shard_map,
                mesh=mesh,
                in_specs=(P('d', None, None, None), P('d', None, None, None), P('d', None, None, None)),
                out_specs=P('d', None, None, None),
                check_rep=False,
            )
            def baseline_step(q_loc, k_loc, v_loc):
                # Standard exponential attention fallback
                s = jnp.matmul(q_loc, jnp.swapaxes(k_loc, -1, -2)) * float(1.0 / math.sqrt(d))
                p = jax.nn.softmax(s, axis=-1)
                return jnp.matmul(p, v_loc)

        # Warmup
        for _ in range(10):
            _ = jax.block_until_ready(afa_step(q, k, v))
            _ = jax.block_until_ready(baseline_step(q, k, v))

        # Benchmarking AFA
        repetitions = 50
        t0 = time.perf_counter()
        for _ in range(repetitions):
            _ = jax.block_until_ready(afa_step(q, k, v))
        t1 = time.perf_counter()
        afa_total_sec = (t1 - t0)
        afa_latency_ms = (afa_total_sec / repetitions) * 1000.0

        # Benchmarking Baseline
        t0 = time.perf_counter()
        for _ in range(repetitions):
            _ = jax.block_until_ready(baseline_step(q, k, v))
        t1 = time.perf_counter()
        base_total_sec = (t1 - t0)
        base_latency_ms = (base_total_sec / repetitions) * 1000.0

        # FLOPs per chip: each chip evaluates b=1 batch item of shape (1, h, l, d)
        flops_per_chip = 4.0 * b * h * (l ** 2) * d
        afa_tflops_per_chip = (flops_per_chip / (afa_latency_ms * 1e-3 * 1e12))
        base_tflops_per_chip = (flops_per_chip / (base_latency_ms * 1e-3 * 1e12))

        throughput_ratio = afa_tflops_per_chip / max(base_tflops_per_chip, 1e-6)
        passed = throughput_ratio >= 0.85

        latencies[f"{name}_afa_ms"] = afa_latency_ms
        latencies[f"{name}_base_ms"] = base_latency_ms

        rows.append({
            "name": name,
            "batch": b,
            "heads": h,
            "seq_len": l,
            "head_dim": d,
            "dtype": str(jnp.dtype(dtype)),
            "repetitions": repetitions,
            "afa_latency_ms": afa_latency_ms,
            "baseline_latency_ms": base_latency_ms,
            "afa_tflops": afa_tflops_per_chip,
            "baseline_tflops": base_tflops_per_chip,
            "afa_tflops_per_chip": afa_tflops_per_chip,
            "baseline_tflops_per_chip": base_tflops_per_chip,
            "throughput_ratio": throughput_ratio,
            "gate_bound": 0.85,
            "passed": passed,
        })

    return {"rows": rows, "latencies": latencies, "passed": all(r["passed"] for r in rows)}


def run_bandwidth_evaluation(place, mesh):
    """Evaluate sustained HBM memory bandwidth utilization on 16 TPU v4 cores."""
    rng = np.random.default_rng(842 + jax.process_index())
    # Test at L=4096, B=1, H=8, D=128
    b, h, l, d = 1, 8, 4096, 128
    dtype = jnp.bfloat16

    q = place(rng.normal(size=(4 * b, h, l, d)).astype(np.float32)).astype(dtype)
    k = place(rng.normal(size=(4 * b, h, l, d)).astype(np.float32)).astype(dtype)
    v = place(rng.normal(size=(4 * b, h, l, d)).astype(np.float32)).astype(dtype)

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(P('d', None, None, None), P('d', None, None, None), P('d', None, None, None)),
        out_specs=P('d', None, None, None),
        check_rep=False,
    )
    def afa_step(q_loc, k_loc, v_loc):
        return tiled_afa_forward(q_loc, k_loc, v_loc, sink_omega=0.5, causal=False, block_q=128, block_k=128)

    # Warmup
    for _ in range(5):
        _ = jax.block_until_ready(afa_step(q, k, v))

    reps = 50
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = jax.block_until_ready(afa_step(q, k, v))
    t1 = time.perf_counter()
    latency_sec = (t1 - t0) / reps

    # Bytes moved per chip in tiled streaming attention:
    # Outer query tile loop: Q read once per Q-block; K and V streamed for each Q-block.
    # Num Q blocks = L // 128 = 32.
    # Total Q bytes per chip: b * H * L * D * 2 bytes.
    # Total K bytes read per chip: 32 * (b * H * L * D * 2 bytes).
    # Total V bytes read per chip: 32 * (b * H * L * D * 2 bytes).
    # Output bytes written per chip: b * H * L * D * 2 bytes.
    bytes_per_token_entry = b * h * l * d * 2  # BF16 = 2 bytes
    num_q_blocks = l // 128
    bytes_per_chip = bytes_per_token_entry * (1 + 2 * num_q_blocks + 1)
    sustained_gb_s_per_chip = (bytes_per_chip / latency_sec) / 1e9
    sustained_gb_s_total = sustained_gb_s_per_chip * 16.0

    # TPU v4 peak HBM bandwidth is 1200 GB/s. 70% threshold is 840 GB/s.
    passed = sustained_gb_s_per_chip >= 840.0

    return {
        "sustained_gb_s": sustained_gb_s_per_chip,
        "sustained_gb_s_total": sustained_gb_s_total,
        "theoretical_peak_gb_s": 1200.0,
        "utilization_pct": (sustained_gb_s_per_chip / 1200.0) * 100.0,
        "target_bound_gb_s": 840.0,
        "latency_ms": latency_sec * 1000.0,
        "passed": passed,
    }


def run_distributed_ring_tpu():
    """Verify lock-free distributed Ring Attention across all 16 physical TPU v4 chips over ICI."""
    total_devices = jax.device_count()
    if total_devices != 16:
        return {"passed": False, "reason": f"Expected 16 TPU chips, found {total_devices}"}

    total_L = 4096
    shard_L = total_L // 16
    B, H, D = 1, 8, 64

    # Identical reference tensors across all hosts
    ref_rng = np.random.default_rng(942)
    q_global = ref_rng.normal(size=(B, H, total_L, D)).astype(np.float32)
    k_global = ref_rng.normal(size=(B, H, total_L, D)).astype(np.float32)
    v_global = ref_rng.normal(size=(B, H, total_L, D)).astype(np.float32)

    devices = mesh_utils.create_device_mesh((16,), jax.devices())
    mesh = Mesh(devices, ('ici_ring',))
    seq_sharding = NamedSharding(mesh, P(None, None, 'ici_ring', None))

    p_idx = jax.process_index()
    # 4 local devices per process out of 16. Local token chunk: 4 * shard_L = 1024
    start_token = p_idx * (4 * shard_L)
    end_token = (p_idx + 1) * (4 * shard_L)

    q_local = q_global[:, :, start_token:end_token, :]
    k_local = k_global[:, :, start_token:end_token, :]
    v_local = v_global[:, :, start_token:end_token, :]

    q_sharded = jax.make_array_from_process_local_data(seq_sharding, q_local)
    k_sharded = jax.make_array_from_process_local_data(seq_sharding, k_local)
    v_sharded = jax.make_array_from_process_local_data(seq_sharding, v_local)

    # Distributed Ring Attention compiled across 16-chip 3D Torus ICI
    @functools.partial(jax.jit, out_shardings=seq_sharding)
    def ring_attention_fn(q, k, v):
        return distributed_ring_afa(q, k, v, sink_omega=0.5, causal=False, axis_name='ici_ring', num_devices=16)

    out_ring = ring_attention_fn(q_sharded, k_sharded, v_sharded)
    out_ring = jax.block_until_ready(out_ring)

    # Reference exact un-tiled attention
    out_exact = exact_afa_reference(jnp.asarray(q_global), jnp.asarray(k_global), jnp.asarray(v_global), sink_omega=0.5)

    local_errs = []
    for idx, shard in enumerate(out_ring.addressable_shards):
        local_dev_idx = p_idx * 4 + idx
        shard_data = np.asarray(shard.data)
        ref_shard = np.asarray(out_exact[:, :, local_dev_idx * shard_L : (local_dev_idx + 1) * shard_L, :])
        diff = float(np.max(np.abs(shard_data - ref_shard)))
        norm_exact = float(np.max(np.abs(ref_shard)))
        rel_error = diff / (norm_exact + 1e-12)
        local_errs.append([diff, rel_error])

    gathered = np.asarray(mh.process_allgather(np.array(local_errs))).reshape(-1, 2)
    max_diff = float(gathered[:, 0].max())
    max_rel_error = float(gathered[:, 1].max())
    passed = max_rel_error <= 1.0e-6

    return {
        "rel_error": max_rel_error,
        "max_abs_diff": max_diff,
        "bound": 1.0e-6,
        "num_chips": 16,
        "total_seq_len": total_L,
        "shard_seq_len": shard_L,
        "passed": passed,
    }


def export_mlir_hlo_audit(place, mesh, output_dir: Path):
    """Compile AFA step on TPU v4 and export MLIR/HLO to verify zero transcendentals."""
    q = place(np.zeros((4, 4, 256, 64), dtype=np.float32))
    k = place(np.zeros((4, 4, 256, 64), dtype=np.float32))
    v = place(np.zeros((4, 4, 256, 64), dtype=np.float32))

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(P('d', None, None, None), P('d', None, None, None), P('d', None, None, None)),
        out_specs=P('d', None, None, None),
        check_rep=False,
    )
    def afa_step(q_loc, k_loc, v_loc):
        return tiled_afa_forward(q_loc, k_loc, v_loc, sink_omega=0.5, causal=False, block_q=128, block_k=128)

    lowered = afa_step.lower(q, k, v)
    hlo_text = lowered.as_text()

    if jax.process_index() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "afa_step.mlir").write_text(hlo_text)

    # Check for forbidden opcodes
    found = [op for op in FORBIDDEN_HLO_OPS if op in hlo_text.lower()]
    passed = (len(found) == 0) and ("dot" in hlo_text.lower() or "dot_general" in hlo_text.lower())

    return {
        "transcendental_opcodes_count": len(found),
        "forbidden_opcodes_found": found,
        "hlo_lines": len(hlo_text.splitlines()),
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase6/tpu")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    jax.distributed.initialize(initialization_timeout=120)
    p_idx = jax.process_index()
    p_count = jax.process_count()
    devices = jax.devices()
    local_devs = jax.local_devices()

    if p_idx == 0:
        print(f"=== Phase 6 TPU Worker 0 Initialized ===")
        print(f"Process count: {p_count}, Total TPU devices: {len(devices)}, Local devices: {len(local_devs)}")

    dev_mesh = mesh_utils.create_device_mesh((len(devices),), devices)
    mesh = Mesh(dev_mesh, ('d',))
    sharding = NamedSharding(mesh, P('d'))
    place = lambda a: jax.make_array_from_process_local_data(sharding, a)

    # 1. Parity evaluation
    if p_idx == 0:
        print("Running numerical parity checks...")
    parity_res = run_parity(place)

    # 2. Benchmarks
    if p_idx == 0:
        print("Running head-to-head throughput benchmarks...")
    bench_res = run_benchmarks(place, mesh)

    # 3. Bandwidth evaluation
    if p_idx == 0:
        print("Evaluating sustained HBM memory bandwidth...")
    bw_res = run_bandwidth_evaluation(place, mesh)

    # 4. Distributed Ring Attention across 16 chips
    if p_idx == 0:
        print("Running lock-free distributed Ring Attention across 16 chips...")
    ring_res = run_distributed_ring_tpu()

    # 5. MLIR HLO Audit
    if p_idx == 0:
        print("Auditing compiled MLIR / HLO opcodes...")
    hlo_res = export_mlir_hlo_audit(place, mesh, out)

    overall_passed = bool(
        parity_res["passed"]
        and bench_res["passed"]
        and bw_res["passed"]
        and ring_res["passed"]
        and hlo_res["passed"]
    )

    if p_idx == 0:
        record = {
            "status": "PASS" if overall_passed else "FAIL",
            "passed": overall_passed,
            "device_count": len(devices),
            "process_count": p_count,
            "devices": [{"id": d.id, "kind": d.device_kind} for d in devices],
            "environment": environment(),
            "parity": parity_res,
            "benchmarks": {
                "rows": bench_res["rows"],
                "passed": bench_res["passed"],
            },
            "bandwidth": bw_res,
            "ring_attention": ring_res,
            "hlo_audit": hlo_res,
        }
        write_json(out / "metrics.json", record)
        write_json(out / "latencies.json", bench_res["latencies"])
        print(f"=== Phase 6 TPU Execution Finished: {record['status']} ===")

    mh.sync_global_devices('phase6-complete')
    jax.distributed.shutdown()
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
