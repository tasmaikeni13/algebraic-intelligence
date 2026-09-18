# Reproduce Phase 4

Run the CPU and formal verification suite from a fresh clone:

```bash
git clone https://github.com/tasmaikeni13/algebraic-intelligence.git
cd algebraic-intelligence
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PATH="$HOME/.elan/bin:$PATH"
(cd formal && lake exe cache get && lake build)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/run_verify_loss.py
```

This verifies the Lean certificates, the zero-transcendental source and graph
audits, strict propriety for random soft targets, the exact probability-domain
gradient, finite gradients through AVN + A-Softmax, Fisher metric equivalence,
same-domain label-noise characterization, and classification convergence.

## Four-host TPU measurements

```bash
.venv/bin/python scripts/launch_phase4_tpu.py --name my-tpu-v4 --zone us-central2-b
.venv/bin/python scripts/run_verify_loss.py
```

The launcher snapshots a clean source commit, runs host tests on every worker,
then executes FP32/BF16 parity and 100 synchronized throughput repetitions for
OACE versus cross-entropy at 256, 1024, 4096, and 32000 classes. It collects the
coordinator record and compiler artifacts in `results/phase4/tpu/`. The final
verifier rejects missing, stale, or source-hash-mismatched TPU evidence.
