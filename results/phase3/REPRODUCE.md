# Reproduce Phase 3

Run these commands in a fresh shell to reproduce the CPU and formal verification suite:

```bash
git clone -b alternative https://github.com/tasmaikeni13/algebraic-intelligence.git
cd algebraic-intelligence
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PATH="$HOME/.elan/bin:$PATH"
(cd formal && lake exe cache get && lake build)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/run_verify_ago.py
```

The full CPU scientific suite verifies:
1. Lean 4 formal certificates in `formal/AlgebraicTheory/Cayley.lean` with 0 warnings/sorry/admit/axioms.
2. Complete test suite in `tests/test_ago.py` and `tests/test_phase3_records.py`.
3. AST, token, and tracer graph audit confirming 0 transcendental operations in `src/attention.py`.
4. Lie group unimodularity and orthogonality ($10^5$ samples, error $\le 4.44 \times 10^{-16}$).
5. Shift equivariance over context $L = 4096$ ($2.1 \times 10^6$ pairs, error $\le 2.10 \times 10^{-14}$).
6. Relative attention dot product parity ($10^5$ pairs, error $\le 2.34 \times 10^{-13}$).
7. Cumulative norm conservation across $m \in [1, 8192]$ (drift $\le 2.22 \times 10^{-16}$).
8. Out-of-distribution associative recall generalization (trained on $L = 256$; tested on $L = 1024$ [100%] and $L = 2048$ [99.5%]).

## Four-host TPU measurements

The companion TPU benchmark is executed across all 16 physical chips of the Cloud TPU v4-32 Pod slice (`my-tpu-v4` in `us-central2-b`):

```bash
.venv/bin/python scripts/launch_phase3_tpu.py --name my-tpu-v4 --zone us-central2-b
.venv/bin/python scripts/run_verify_ago.py
```

The distributed launcher:
1. Checks that accelerator devices are idle via `sudo fuser -s /dev/accel*`.
2. Bundles the clean git commit snapshot and transfers it across all 4 worker nodes via SCP.
3. Sets up independent virtual environments and verifies all CPU host tests before initializing JAX distributed runtime.
4. Executes `scripts/run_phase3_tpu.py` across 16 TPU v4 cores.
5. Verifies TPU numerical parity (20 configurations in FP32 and BF16), unimodularity, shift equivariance, and norm conservation.
6. Times 100 synchronized repetitions per configuration for AGO vs RoPE with interleaved randomized ordering and multihost barriers across lengths 512, 1024, 2048, and 4096 in FP32 and BF16.
7. Collects coordinator measurements and HLO/MLIR graphs into `results/phase3/tpu/`.
