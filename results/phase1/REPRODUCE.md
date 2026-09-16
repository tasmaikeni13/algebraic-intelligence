# Reproduce Phase 1

## Fresh host setup and CPU/formal verification

Run from a fresh shell. A Python 3.10 environment and the committed Lean toolchain
are used. On Ubuntu, install `python3.10-venv` if `python3 -m venv` reports that
ensurepip is missing. Install elan using its official installer if absent.

```bash
git clone https://github.com/tasmaikeni13/algebraic-intelligence.git
cd algebraic-intelligence
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PATH="$HOME/.elan/bin:$PATH"
(cd formal && lake exe cache get && lake build)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_verify_primitives.py --cpu-only
```

`--cpu-only` runs all 10,000 trials at each of four depths and all million-sample
scale experiments. It returns success for the CPU subset and explicitly records
Phase 1 as incomplete until current TPU evidence is present. Without this flag,
missing hardware evidence makes the command exit nonzero.

## Four-host TPU verification

The existing node `my-tpu-v4` in `us-central2-b` is Cloud TPU **v4-32**: 16 physical
v4 chips, four hosts, physical topology 2×2×4. No new node is provisioned. The
launcher checks that devices are free and exits without stopping other jobs.
It bundles the committed source, makes isolated snapshot directories on all four
hosts, installs pinned dependencies, runs CPU tests on each host, then starts the
same JAX distributed program on every host.

```bash
.venv/bin/python scripts/launch_phase1_tpu.py --name my-tpu-v4 --zone us-central2-b
.venv/bin/python scripts/run_verify_primitives.py
```

The second command validates TPU source hashes, the complete device inventory,
float32/BF16 numerical parity, Monte Carlo identities, all 40,000 deep trials,
and ALU/GELU and AVN/RMSNorm forward and forward+backward throughput thresholds.
Timing includes `jax.block_until_ready`, excludes compilation/transfers, uses
10 warmups and 100 randomized-order repetitions, and reports the slowest host's
latency in each sample. RMSNorm/LayerNorm receive dynamic learnable scale arrays;
their backward timing includes parameter cotangents. Swish and LayerNorm are
additional measured comparison arms. Raw timing samples and trials are retained.

For another already-provisioned matching slice, supply its name and zone. The
program rejects other device counts or accelerator generations. It never counts
CPU or emulated-device timings as TPU measurements.

## Evidence and limits

See `metrics.json`, `STATUS.md`, `ITERATIONS.md`, and `DEPENDENCIES.md`. A Phase 1
PASS record is written only after every required CPU/formal and TPU gate passes
and the artifacts have been audited. All test-network claims specify width 128,
He initialization, residual attenuation, seeds, and sample counts. Phase 1 does
not establish trainability or performance for later transformer implementations.

## Revalidate preserved CPU samples after a metadata-only change

```bash
.venv/bin/python scripts/run_verify_primitives.py --reuse-cpu results/phase1/iterations/full-cpu-width128/metrics.json
```

The verifier rejects changed numerical dependencies or numerical package versions;
it always rebuilds Lean, runs the unit suite and purity audit, and checks current
TPU evidence. Omit `--reuse-cpu` to repeat every CPU sample. TPU output is collected
from all four hosts because cloud worker order need not equal JAX process order.
