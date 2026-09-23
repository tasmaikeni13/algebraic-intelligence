# Phase 6 PASS — Pallas Algebraic FlashAttention on 16 TPU v4 Chips

Verified 2026-09-23 on CPU and the four-host, 16-chip Cloud TPU v4 Pod slice
`my-tpu-v4` in `us-central2-b`. TPU parity, benchmarking, and compiler auditing
all exercise the real `pallas_afa_forward` implementation. The throughput
baseline is JAX's Pallas TPU FlashAttention kernel, not dense attention.

## Gate inventory

| Gate | Verified outcome | Evidence |
| :--- | :--- | :--- |
| Lean certificates | 1527 jobs; no errors, `sorry`, `admit`, or added axioms | [`lean-build.log`](lean-build.log), [`Kernel.lean`](../../formal/AlgebraicTheory/Kernel.lean), [`Gate.lean`](../../formal/AlgebraicTheory/Gate.lean) |
| Zero-transcendental source audit | No source, token, or traced-graph violations | [`metrics.json`](metrics.json) |
| CPU tiled accuracy | Maximum float64 relative error $1.50\times10^{-15}$ over 12 configurations | [`metrics.json`](metrics.json) |
| Additive invariants | Maximum block-size drift $4.44\times10^{-15}$; scale-invariance drift $4.44\times10^{-16}$ | [`metrics.json`](metrics.json) |
| Real Pallas identity | Parity, benchmark, and HLO rows identify `pallas_afa_forward`; source hashes match the current implementation | [`tpu/metrics.json`](tpu/metrics.json) |
| TPU numerical parity | 8/8 causal/non-causal FP32/BF16 configurations passed; all finite | [`tpu/metrics.json`](tpu/metrics.json) |
| Pallas throughput | Ratios $0.9883$ at $L=2048$ and $0.9986$ at $L=4096$ versus `jax.experimental.pallas.ops.tpu.flash_attention` | [`tpu/metrics.json`](tpu/metrics.json), [`tpu/latencies.json`](tpu/latencies.json) |
| Bounded streaming storage | Conservative tile working set 294,912 bytes (288 KiB), within the 16 MiB VMEM budget; no $L\times L$ materialization | [`tpu/metrics.json`](tpu/metrics.json) |
| Distributed ring parity | Relative error $4.15\times10^{-7}$ across 16 chips; maximum absolute difference $9.54\times10^{-7}$ | [`tpu/metrics.json`](tpu/metrics.json) |
| TPU compiler audit | 0 forbidden transcendental opcodes; `tpu_custom_call` Pallas lowering marker present | [`tpu/metrics.json`](tpu/metrics.json), [`tpu/afa_step.mlir`](tpu/afa_step.mlir) |

## Real Pallas numerical parity

The real Pallas kernel was compared with an independent float64 CPU oracle.

| Shape $(B,H,L,D)$ | Causal | Dtype | Maximum error | Tolerance |
| :---: | :---: | :---: | ---: | ---: |
| $(1,4,256,64)$ | No | FP32 | $6.614\times10^{-7}$ | $2\times10^{-4}$ |
| $(1,4,256,64)$ | Yes | FP32 | $6.093\times10^{-7}$ | $2\times10^{-4}$ |
| $(1,8,512,64)$ | No | FP32 | $5.546\times10^{-7}$ | $2\times10^{-4}$ |
| $(1,8,512,64)$ | Yes | FP32 | $7.246\times10^{-7}$ | $2\times10^{-4}$ |
| $(1,8,1024,64)$ | No | BF16 | $5.142\times10^{-3}$ | $4\times10^{-2}$ |
| $(1,8,1024,64)$ | Yes | BF16 | $6.143\times10^{-3}$ | $4\times10^{-2}$ |
| $(1,8,2048,64)$ | No | BF16 | $5.320\times10^{-3}$ | $4\times10^{-2}$ |
| $(1,8,2048,64)$ | Yes | BF16 | $5.036\times10^{-3}$ | $4\times10^{-2}$ |

FP32 Pallas matrix products use high precision with FP32 accumulation. BF16
uses TPU-legal default operand precision with FP32 accumulation. The row-sum
scratch retains a TPU-native replicated 128-lane layout and normalization is
performed inside the kernel.

## Pallas-vs-Pallas throughput

Each row used 10 warmups and 50 synchronized repetitions across all 16 chips.

| Sequence | AFA latency (ms) | Baseline latency (ms) | AFA TFLOPS/chip | Baseline TFLOPS/chip | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1.072545 | 1.059955 | 16.0179 | 16.2081 | 0.988262 |
| 4096 | 3.204489 | 3.199845 | 21.4448 | 21.4759 | 0.998551 |

The baseline is `jax.experimental.pallas.ops.tpu.flash_attention`. Both rows
clear the $\ge85\%$ throughput gate.

## Streaming and distributed contracts

The conservative live tile set (Q, K, V, score, numerator, and replicated
denominator scratch) is 288 KiB, well below the 16 MiB VMEM budget. The
$10.785\,\mathrm{GB/s}$ figure in the machine-readable record is only a minimum
logical external-I/O rate derived from tensor sizes and latency. It is not a
physical HBM-utilization claim. No bandwidth percentage is inferred without
hardware profiler counters.

For the 16-chip ring check, $L=4096$ is divided into 16 shards of 256 tokens.
Numerators and denominators combine additively before one final normalization,
with no running-max or exponential rescaling step.

The machine-readable JSON files are authoritative for unrounded values, exact
implementation identities, and source hashes.
