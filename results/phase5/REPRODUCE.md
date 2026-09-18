# Reproduce Phase 5

Run the CPU and formal verification suite from a fresh clone:

```bash
git clone https://github.com/tasmaikeni13/algebraic-intelligence.git
cd algebraic-intelligence
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PATH="$HOME/.elan/bin:$PATH"
(cd formal && lake exe cache get && lake build)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/run_verify_optimizer.py
```

This verifies the Lean certificates, zero-transcendental audits, exact default
Optax semantics, ill-conditioned and non-convex experiments, ARDS monotonicity
and asymptotics, and optimizer-parameter isolation.

## Four-host TPU measurements

```bash
.venv/bin/python scripts/launch_phase5_tpu.py --name my-tpu-v4 --zone us-central2-b
.venv/bin/python scripts/run_verify_optimizer.py
```

The launcher snapshots a clean source commit, runs host tests on every worker,
then executes FP32/BF16 numerical parity, the quadratic sweep, and 100
synchronized repetitions of ARDS versus cosine scheduling. It also audits the
compiled optimizer-step MLIR: the ARDS path must contain `rsqrt`, contain no raw
`sqrt`, and match the implementation whose source hashes are recorded. The
final verifier rejects missing, stale, or source-hash-mismatched TPU evidence.
