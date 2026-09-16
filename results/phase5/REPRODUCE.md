# Reproduce Phase 5

Run these commands in a fresh shell to reproduce the CPU and formal verification suite:

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

The full CPU scientific suite verifies:
1. Lean 4 formal certificates in `formal/AlgebraicTheory/Curvature.lean` with 0 warnings/sorry/admit/axioms.
2. Complete test suite in `tests/test_optimizer.py` and `tests/test_phase5_records.py` (112 unit and property tests).
3. AST, token, and tracer graph audit confirming 0 transcendental operations in `src/optimizer.py`.
4. Ill-conditioned quadratic surface optimization sweep ($10^4$ trials, condition numbers $\kappa \in [10^2, 10^6]$, $> 99.99\%$ loss reduction in 300 steps).
5. Non-convex stochastic surface benchmarks (Rosenbrock and Rastrigin with noise $\sigma = 0.5$ across 200 random seeds, ARDS matches Cosine Annealing within $\le 2\%$).
6. ARDS schedule monotonicity across $10^5$ steps and asymptotic $\mathcal{O}(1/t)$ convergence rate verification.
7. Architectural isolation contract verifying exact optimizer parameter parity ($\beta_1, \beta_2, \epsilon, \lambda$) between algebraic and baseline models.

## Four-host TPU measurements

The companion TPU benchmark is executed across all 16 physical chips of the Cloud TPU v4-32 Pod slice (`my-tpu-v4` in `us-central2-b`):

```bash
.venv/bin/python scripts/launch_phase5_tpu.py --name my-tpu-v4 --zone us-central2-b
.venv/bin/python scripts/run_verify_optimizer.py
```

The distributed launcher:
1. Checks that accelerator devices are idle via `sudo fuser -s /dev/accel*`.
2. Bundles the clean git commit snapshot and transfers it across all 4 worker nodes via SCP.
3. Sets up independent virtual environments and verifies all CPU host tests before initializing JAX distributed runtime.
4. Executes `scripts/run_phase5_tpu.py` across 16 TPU v4 cores.
5. Verifies TPU numerical parity against float64 CPU oracle across FP32 and BF16 shapes, 0 NaNs, 0 Infs.
6. Verifies ill-conditioned quadratic optimization sweep on TPU hardware ($10^4$ trials).
7. Times 100 synchronized repetitions per configuration for Algebraic AdamW + ARDS vs AdamW + Cosine Annealing with interleaved randomized ordering and multihost barriers across parameter matrices in FP32 and BF16.
8. Collects coordinator measurements and HLO/MLIR graphs into `results/phase5/tpu/`.
