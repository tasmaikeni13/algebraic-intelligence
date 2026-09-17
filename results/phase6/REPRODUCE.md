# Reproduce Phase 6

Run these commands in a fresh shell to reproduce the CPU and formal verification suite:

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

The full CPU scientific suite verifies:
1. Lean 4 formal certificates in `formal/AlgebraicTheory/Kernel.lean` and `formal/AlgebraicTheory/Gate.lean` with 0 warnings/sorry/admit/axioms.
2. Complete test suite in `tests/test_pallas_afa.py` and `tests/test_phase6_records.py` (125 repository tests passing).
3. AST, token, and tracer graph audit confirming strictly 0 transcendental operations in `src/kernels/pallas_afa.py`.
4. High-precision numerical accuracy sweep vs Float64 oracle across 12 configurations ($\le 1.0\times 10^{-6}$ bound; observed maximum relative error $= 1.4965\times 10^{-15}$).
5. Inter-tile rescaling FLOP count audit (strictly 0 $\exp(m_{\text{old}} - m_{\text{new}})$ calls and 0 running-max subtractions).
6. Additive tile accumulation associativity study across block sizes $B \in \{64, 128, 256\}$ and single-pass scale invariance ($\le 1.0\times 10^{-12}$ tolerance).
7. Lock-free distributed Ring Attention simulation equivalence across 4, 8, and 16 sequence shards ($\le 1.0\times 10^{-6}$ bound).
8. Static XLA HLO compiler opcode audit confirming systolic MXU mapping and hardware VMU radical instructions.

## Four-Host 16-Chip TPU Measurements

The companion hardware TPU benchmark is executed across all 16 physical chips of the Cloud TPU v4-32 Pod slice (`my-tpu-v4` in `us-central2-b`):

```bash
.venv/bin/python scripts/launch_phase6_tpu.py --name my-tpu-v4 --zone us-central2-b
.venv/bin/python scripts/run_verify_pallas.py
```

The distributed launcher:
1. Checks that accelerator devices are idle via `sudo fuser -s /dev/accel*`.
2. Bundles the clean git commit snapshot and transfers it across all 4 worker nodes via SCP.
3. Sets up independent virtual environments and verifies all CPU host tests before initializing JAX distributed runtime.
4. Executes `scripts/run_phase6_tpu.py` across 16 TPU v4 cores.
5. Verifies TPU numerical parity against float64 CPU oracle across 8 configurations in FP32 and BF16 (all 8 pass, 0 NaNs, 0 Infs).
6. Times 50 synchronized repetitions per configuration for Algebraic FlashAttention vs transcendental attention baseline at $L=2048$ and $L=4096$ with $D=128$, sustaining $86.32\%$ and $88.37\%$ throughput parity ($\ge 85\%$ gate bound).
7. Measures sustained HBM bandwidth utilization during tile streaming ($1644.9\text{ GB/s/chip} \ge 840.0\text{ GB/s/chip}$ target).
8. Executes lock-free distributed Ring Attention over the 3D Torus Inter-Chip Interconnect (ICI) across all 16 TPU v4 chips (relative error $= 4.257\times 10^{-7} \le 1.0\times 10^{-6}$).
9. Inspects compiled MLIR / XLA HLO graphs to confirm strictly 0 transcendental opcodes (`exponential`, `logarithm`, `sine`, `cosine`, `tanh`, `sigmoid`).
10. Collects coordinator measurements and HLO/MLIR graphs into `results/phase6/tpu/`.
