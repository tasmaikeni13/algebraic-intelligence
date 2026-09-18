# Reproduce Phase 6

Run the CPU and formal verification suite from a fresh clone:

```bash
git clone https://github.com/tasmaikeni13/algebraic-intelligence.git
cd algebraic-intelligence
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PATH="$HOME/.elan/bin:$PATH"
(cd formal && lake exe cache get && lake build)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/run_verify_pallas.py
```

This verifies the Lean certificates, zero-transcendental audits, float64 tiled
accuracy, additive accumulation invariants, and static HLO properties.

## Four-host TPU measurements

```bash
.venv/bin/python scripts/launch_phase6_tpu.py --name my-tpu-v4 --zone us-central2-b
.venv/bin/python scripts/run_verify_pallas.py
```

The launcher snapshots a clean source commit, runs host tests on every worker,
and then:

1. Runs eight causal/non-causal FP32/BF16 parity configurations through the
   real `pallas_afa_forward` kernel.
2. Times 50 synchronized repetitions at $L=2048$ and $L=4096$ against
   `jax.experimental.pallas.ops.tpu.flash_attention`.
3. Verifies the conservative tile working set fits the 16 MiB VMEM budget. It
   does not claim physical HBM utilization without profiler counters.
4. Checks 16-chip additive ring equivalence.
5. Audits the compiled Pallas artifact for forbidden transcendental opcodes and
   a TPU custom-call lowering marker.

Artifacts are collected in `results/phase6/tpu/`. The final verifier rejects
fallback implementations, dense baselines, missing compiler evidence, stale
records, or source-hash mismatches.
