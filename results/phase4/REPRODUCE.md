# Reproduce Phase 4

Run these commands in a fresh shell to reproduce the CPU and formal verification suite:

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

The full CPU scientific suite verifies:
1. Lean 4 formal certificates in `formal/AlgebraicTheory/Loss.lean` with 0 warnings/sorry/admit/axioms.
2. Complete test suite in `tests/test_loss.py` and `tests/test_phase4_records.py`.
3. AST, token, and tracer graph audit confirming 0 transcendental operations in `src/loss.py`.
4. Monte Carlo label noise gradient variance study ($10^5$ trials, $\operatorname{Var}(\nabla \mathcal{L}_{1/8}) / \operatorname{Var}(\nabla \mathcal{L}_{\text{CE}}) \le 0.50$).
5. Simplex boundary stability down to $p_k = 10^{-9}$ (gradient bounded at $106.68 \le 107.0$, 0 NaNs, 0 Infs).
6. Fisher Information Metric equivalence ($H(D_P) = 2.0 \cdot H(D_{\text{KL}})$ with relative error $\le 10^{-11}$).
7. Strict propriety and monotonicity ($\min = 0.0$ at $p_k = 1.0$, strictly negative derivative $\forall p_k \in (0, 1)$).
8. Synthetic classification convergence benchmark (OACE achieves parity with cross-entropy with lower gradient variance).

## Four-host TPU measurements

The companion TPU benchmark is executed across all 16 physical chips of the Cloud TPU v4-32 Pod slice (`my-tpu-v4` in `us-central2-b`):

```bash
.venv/bin/python scripts/launch_phase4_tpu.py --name my-tpu-v4 --zone us-central2-b
.venv/bin/python scripts/run_verify_loss.py
```

The distributed launcher:
1. Checks that accelerator devices are idle via `sudo fuser -s /dev/accel*`.
2. Bundles the clean git commit snapshot and transfers it across all 4 worker nodes via SCP.
3. Sets up independent virtual environments and verifies all CPU host tests before initializing JAX distributed runtime.
4. Executes `scripts/run_phase4_tpu.py` across 16 TPU v4 cores.
5. Verifies TPU numerical parity against float64 CPU oracle, simplex boundary stability, and Fisher equivalence on TPU hardware.
6. Times 100 synchronized repetitions per configuration for OACE vs Cross-Entropy with interleaved randomized ordering and multihost barriers across vocabularies (1024, 4096, 16384, 32768) in FP32 and BF16.
7. Collects coordinator measurements and HLO/MLIR graphs into `results/phase4/tpu/`.
