# Phase 6: Hardware-Fused Kernels & Algebraic FlashAttention (AFA on 16 TPU v4 Pod via Pallas / XLA)

Start only after Phase 5 PASS. Read `theory.md`, official Google Cloud TPU v4 and JAX Pallas documentation, and `phases/README.md`. Execute the adaptive failure-repair loop until PASS.

---

## 1. Objective, Scientific Hypothesis & Competing Models

Eliminate inter-tile synchronization barriers and transcendental online rescaling of FlashAttention on TPU v4 systolic hardware:
$$\textbf{"Can Algebraic FlashAttention achieve near-roofline memory bandwidth on Google Cloud TPU v4 hardware via JAX Pallas?"}$$

### Competing Hypotheses:
- **$H_1$ (Algebraic Hypothesis):** AFA replaces running maximum subtraction $\exp(m_{\text{old}} - m_{\text{new}})$ with pure additive tile accumulation. When implemented in JAX Pallas targeting TPU v4 TensorCore Vector Memory (VMEM) and 128×128 Matrix Multiply Units (MXUs), AFA streams tiles without inter-tile log-sum-exp synchronization barriers, sustaining $> 70\%$ peak HBM bandwidth ($> 840\text{ GB/s}$ per chip) and enabling lock-free distributed Ring Attention across the 16 TPU v4 chips over the 3D Torus Inter-Chip Interconnect (ICI).
- **$H_0$ (Transcendental Baseline Hypothesis):** Standard FlashAttention-2 online exponential rescaling is uniquely optimal for hardware scratchpad caching; additive algebraic kernels will encounter VMEM pressure or numerical overflow during long sequence tile streaming.

---

## 2. Hardware Execution Model & Mathematical Formulations

### 2.1 Pure Additive Tile Accumulation on TPU v4 (Pallas)
For query block $\mathbf{Q}_b \in \mathbb{R}^{B_q \times d}$ and key-value blocks $\mathbf{K}_c, \mathbf{V}_c \in \mathbb{R}^{B_k \times d}$ loaded into TensorCore Vector Memory (VMEM):
1. Raw algebraic scores computed on systolic MXU: $\mathbf{S}_{bc} = \frac{\mathbf{Q}_b \mathbf{K}_c^\top}{\sqrt{d_k}}$.
2. Octic algebraic kernel evaluated on VMU via 3 squaring stages: $\mathbf{P}_{bc} = (\mathbf{S}_{bc} + \sqrt{1 + \mathbf{S}_{bc}^{\odot 2}})^8$.
3. Pure additive accumulation:
   $$\mathbf{O}_b = \sum_{c} \mathbf{P}_{bc} \mathbf{V}_c, \qquad \mathbf{D}_b = \Omega + \sum_{c} \sum_{j} \mathbf{P}_{bc, \cdot j}$$
4. Single-pass tile normalization: $\mathbf{Y}_b = \mathbf{O}_b / \mathbf{D}_b$. Zero inter-tile sync!

### 2.2 TPU v4 Architectural Parameters & Tiling
- **Matrix Units (MXU):** Dual 128×128 systolic matrix multipliers per TensorCore. Block sizes $(B_q, B_k) = (128, 128)$ align natively with the systolic array dimensions to achieve peak compute density.
- **Vector Memory (VMEM):** 16 MB scratchpad memory per TensorCore. Tile sizing guarantees zero spillover to HBM.
- **Distributed Ring Attention over ICI:** Single global AllReduce sum across the 16 TPU v4 chips via `jax.lax.psum` over the 3D Torus ICI interconnect, with zero intermediate log-sum-exp tile synchronization.

---

## 3. Implementation Target: JAX Pallas / TPU v4 Architecture

Instruct the creation and verification of the following files targeting the 16 TPU v4 Pod:
1. **`src/kernels/pallas_afa.py`**:
   - Native JAX Pallas TPU kernel using `jax.experimental.pallas.tpu` (`pallas_call`).
   - Pure additive tile accumulation in VMEM with zero exponential rescaling.
   - Forward and backward kernel passes optimized for BF16 execution on TPU v4 MXUs and VMUs.
   - Distributed Ring Attention wrapper leveraging TPU v4 ICI interconnect.
2. **`tests/test_kernel_parity.py`**:
   - Parity verification comparing Pallas TPU AFA against an un-tiled float64 CPU reference.
   - Numerical error tolerance: relative error $\le 1.0 \times 10^{-6}$.
3. **`scripts/run_benchmark_pallas.py`**:
   - Benchmarking harness measuring TFLOPS throughput and HBM bandwidth across sequence lengths $L \in [1024, 8192]$ on 16 TPU v4 chips.
   - Static XLA HLO / compiler inspection verifying exactly zero transcendental opcodes (`exp`, `log`).

---

## 4. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Kernel.lean` and `formal/AlgebraicTheory/Gate.lean` under `/root/.elan/bin/lake build`:
1. Single-pass additive associativity: $\sum (P_1 V_1 + P_2 V_2) = (\sum P_1 V_1) + (\sum P_2 V_2)$.
2. Numerator-denominator scaling invariance: $(\alpha O) / (\alpha D) = O / D$ for $\alpha > 0$.

---

## 5. Hardware Benchmarking & Passing Gate on 16 TPU v4 Pod

Benchmark `src/kernels/pallas_afa.py` via `scripts/run_benchmark_pallas.py` on the 16 TPU v4 Pod:

| Evaluation Dimension | Target on 16 TPU v4 Pod | Tolerance / Bound |
| :--- | :--- | :--- |
| **Numerical Accuracy vs. Float64** | $\|\mathbf{Y}_{\text{AFA}} - \mathbf{Y}_{\text{exact}}\|_\infty / \|\mathbf{Y}_{\text{exact}}\|_\infty$ | $\leq 1.0 \times 10^{-6}$ |
| **Inter-Tile Rescaling FLOPs** | Transcendental $\exp(m_{\text{old}} - m_{\text{new}})$ calls in AFA | Exactly $0$ |
| **Kernel Throughput at $L=4096$** | TFLOPS per chip (BF16 forward) | $\geq 85\%$ of baseline FlashAttention-2 |
| **HBM Memory Bandwidth Utilization** | Sustained GB/s during tile streaming | $\geq 70\%$ of theoretical peak ($> 840\text{ GB/s}$/chip) |
| **Distributed Ring Attention Relative Error** | 16 TPU v4 chips, additive accumulation error | $\leq 1.0 \times 10^{-6}$ |
| **Zero Transcendental Audit** | XLA HLO / AST inspection | Exactly $0$ transcendental math opcodes |

---

## 6. Adaptive Failure-Repair & Bidirectional Dependency Protocol

When a test or gate fails in Phase 6:
1. **Iterate Locally:**
   - If VMEM allocation exceeds scratchpad capacity, tune tile dimensions from $(128, 128)$ to $(128, 64)$ or adjust Pallas memory allocation hints.
   - If compilation errors occur in Pallas TPU compiler, inspect grid dimensions and block spec mappings.
2. **Backward Rollback to Phase 2/1:**
   - If numerical instability or accumulator overflow occurs during long sequence streaming, inspect Phase 2 A-Softmax logit scaling $\tau$ or attention sink $\Omega$.
   - If Phase 2 A-Softmax or Phase 1 AVN must be modified, backtrack to Phase 2 (or Phase 1), update the core primitives, re-prove Lean 4 theorems, pass their regression gates, and forward-cascade the updates back to Phase 6.
3. **Forward Dependency Cascading:**
   - **Phases 7, 8, 9 (`src/model.py`, pretraining pipelines):** Integrate `pallas_afa` as the attention forward/backward engine in `AlgebraicTransformerLM`. Any changes to tile signatures or sharding constraints must be propagated into `src/model.py` and `src/mesh.py`.

---

## 7. PASS Gates

- [ ] `formal/AlgebraicTheory/Kernel.lean` compiles with 0 errors via `/root/.elan/bin/lake build`.
- [ ] `src/kernels/pallas_afa.py` created with JAX Pallas TPU implementation for TPU v4 VMU/MXU.
- [ ] Relative numerical accuracy against float64 un-tiled reference is $\le 1.0 \times 10^{-6}$.
- [ ] Head-to-head throughput benchmark executed against FlashAttention-2 on 16 TPU v4 Pod.
- [ ] Sustained HBM bandwidth exceeds $70\%$ of theoretical peak ($> 840\text{ GB/s}$ per chip).
- [ ] Distributed Ring Attention executes across 16 TPU v4 chips over ICI with relative error $\le 1.0 \times 10^{-6}$.
- [ ] XLA compiler dump confirms zero transcendental library calls.
- [ ] `results/phase6/PASS.md` satisfies the shared PASS record contract.
