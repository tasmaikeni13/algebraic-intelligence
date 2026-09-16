# Reproduce Phase 2

Run these commands in a fresh shell. The verifier returns exit 1 for the retained
failed scientific gates; a failed experiment is not an installation error.

```bash
git clone https://github.com/tasmaikeni13/algebraic-intelligence.git
cd algebraic-intelligence
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PATH="$HOME/.elan/bin:$PATH"
(cd formal && lake exe cache get && lake build)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/phase2_counterexamples.py
.venv/bin/python scripts/run_verify_attention.py
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python results/phase2/iterations/scale-sink-sweep.py
```

The full CPU run evaluates 100,000 independent score vectors over lengths
64/128/256/512/1024/2048/4096 and 10,000 complete autograd Jacobian matrices over
lengths 2/8/16/64/128. Raw per-trial diagnostics are saved in NPZ files. Random
seeds, counts, definitions, source hashes and numerical package versions are
recorded. The E2M1 diagnostic uses the finite FP4 codebook from the
[OCP MX v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
with explicitly chosen per-vector max-abs/6 scaling and ties toward lower
magnitude. This is software quantization of scores, not native TPU FP4 compute
or a complete implementation of OCP block microscaling.

## Four-host TPU measurements

The existing `my-tpu-v4` in project `projectalgebraicai`, zone `us-central2-b`,
is a 16-physical-chip Cloud TPU v4-32 slice. gcloud authentication and SSH access
are required; no new resources are provisioned. The launcher checks availability,
copies a committed snapshot, builds independent environments and executes CPU
unit tests on every host before initializing JAX distributed execution.

```bash
.venv/bin/python scripts/launch_phase2_tpu.py --name my-tpu-v4 --zone us-central2-b
.venv/bin/python scripts/run_verify_attention.py
```

The launcher and verifier return nonzero when the scientific gates fail. The
launcher still collects measurement files in its `finally` handler. All four
cloud workers are searched because their numbering differs from JAX process
numbering. Each run uses a unique empty measurement directory to exclude stale
bundled evidence. Snapshot commit and remote directory are in `tpu/snapshot.txt`.

Hardware execution covers float32/BF16 oracle and VJP parity; forward and
forward+backward throughput versus `jax.nn.softmax` at lengths 128–4096; 100,352
independent float32 score vectors; and 10,240 full Jacobians. All benchmark arms
use the same inputs/cotangents and 16,777,216 global elements. Timings exclude
compilation, transfers and host barriers, include device synchronization, use
10 warmups and 100 randomized-order repetitions, and retain the slowest host's
latency per repetition. Raw samples, environment and source fingerprints are
published. No CPU timing is used as TPU evidence.

## Inherited Phase 1 gates

```bash
.venv/bin/python scripts/run_verify_primitives.py --reuse-cpu results/phase1/iterations/full-cpu-width128/metrics.json
```

This rebuilds Lean, reruns all tests and primitive purity checks, checks that the
CPU numerical dependencies remain unchanged, and validates current Phase 1 TPU
source hashes and the complete gate inventory. Use the full commands in
`results/phase1/REPRODUCE.md` to repeat every inherited CPU/TPU measurement.
