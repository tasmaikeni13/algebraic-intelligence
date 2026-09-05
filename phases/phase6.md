# Phase 6: Hardware-Fused Kernels & Algebraic FlashAttention (AFA on MI300X)

Start only after Phase 5 PASS. Read `theory.md`, current official AMD ROCm / MI300X documentation, and `phases/AUTONOMY_PROTOCOL.md`. Execute the failure-repair loop until PASS.

---

## 1. Objective, Scientific Hypothesis & Competing Models

Eliminate the inter-tile synchronization barriers and transcendental online rescaling of FlashAttention:
$$\textbf{"Can Algebraic FlashAttention achieve near-roofline memory bandwidth on the AMD Instinct MI300X?"}$$

### Competing Hypotheses:
- **$H_1$ (Algebraic Hypothesis):** AFA replaces running maximum subtraction $\exp(m_{\text{old}} - m_{\text{new}})$ with pure additive tile accumulation, executing within CDNA3 Wave64 vector registers without inter-tile synchronization barriers, sustaining $> 3.5\text{ TB/s}$ HBM3 bandwidth on the dedicated 1x AMD Instinct MI300X GPU (`gfx942`).
- **$H_0$ (Transcendental Baseline Hypothesis):** FlashAttention-2 online exponential rescaling is optimal for GPU SRAM caching; additive algebraic kernels will encounter vector register pressure or numerical overflow during long sequence tile streaming.

---

## 2. Hardware Execution Model & Mathematical Formulations

### 2.1 Pure Additive Tile Accumulation on CDNA3 (MI300X)
For query block $\mathbf{Q}_b \in \mathbb{R}^{B_q \times d}$ and key-value blocks $\mathbf{K}_c, \mathbf{V}_c \in \mathbb{R}^{B_k \times d}$:
1. Raw algebraic scores in Local Data Share (LDS): $\mathbf{S}_{bc} = \frac{\mathbf{Q}_b \mathbf{K}_c^\top}{\sqrt{d_k}}$.
2. Octic algebraic kernel in Wave64 registers via 3 squaring operations: $\mathbf{P}_{bc} = (\mathbf{S}_{bc} + \sqrt{1 + \mathbf{S}_{bc}^{\odot 2}})^8$.
3. Pure additive accumulation:
   $$\mathbf{O}_b = \sum_{c} \mathbf{P}_{bc} \mathbf{V}_c, \qquad \mathbf{D}_b = \Omega + \sum_{c} \sum_{j} \mathbf{P}_{bc, \cdot j}$$
4. Single-pass tile normalization: $\mathbf{Y}_b = \mathbf{O}_b / \mathbf{D}_b$. Zero inter-tile sync!

### 2.2 CDNA3 Architectural Parameters
- **Wavefront:** Native Wave64 (`warp_size = 64`).
- **Peak Bandwidth:** $5.3\text{ TB/s}$ HBM3 on single socket.
- **Register Budget:** 256 Vector General-Purpose Registers (VGPRs) per work-item to eliminate scratch memory spilling.
- **Ring Attention:** Single global AllReduce sum across distributed nodes with zero intermediate tile communication.

---

## 3. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Kernel.lean` and `formal/AlgebraicTheory/Gate.lean` under `/root/.elan/bin/lake build`:
1. Single-pass additive associativity: $\sum (P_1 V_1 + P_2 V_2) = (\sum P_1 V_1) + (\sum P_2 V_2)$.
2. Numerator-denominator scaling invariance: $(\alpha O) / (\alpha D) = O / D$ for $\alpha > 0$.

---

## 4. Hardware Benchmarking & Passing Gate on MI300X

Benchmark `analysis/kernels/algebraic_attention_hip.cpp` and `analysis/kernels/algebraic_attention_triton.py` on the MI300X:

| Evaluation Dimension | Target on MI300X | Tolerance / Bound |
| :--- | :--- | :--- |
| **Numerical Accuracy vs. Float64** | $\|\mathbf{Y}_{\text{AFA}} - \mathbf{Y}_{\text{exact}}\|_\infty / \|\mathbf{Y}_{\text{exact}}\|_\infty$ | $\leq 1.0 \times 10^{-6}$ |
| **Inter-Tile Rescaling FLOPs** | Transcendental $\exp(m_{\text{old}} - m_{\text{new}})$ calls in AFA | Exactly $0$ |
| **Kernel Throughput at $L=4096$** | TFLOPS on MI300X (BF16 forward) | $\geq 85\%$ of baseline FlashAttention-2 |
| **HBM Memory Bandwidth Utilization** | Sustained GB/s during tile streaming | $\geq 3.5\text{ TB/s}$ ($> 65\%$ of theoretical peak) |
| **Distributed Ring Attention Relative Error** | 8 simulated nodes, additive accumulation error | $\leq 1.0 \times 10^{-6}$ |
| **Zero Transcendental Audit** | ISA inspection via `llvm-objdump -d` on compiled `.hsaco` | Exactly $0$ transcendental math opcodes |

---

## 5. Autonomous Failure Ledger & Self-Correction Playbook

- **Symptom: High VGPR pressure spills to scratch memory on MI300X:**
  - *Root Cause:* Block size $(B_q, B_k) = (128, 128)$ with unrolled squaring exceeds register capacity.
  - *Correction:* Tune tile dimensions to $(64, 64)$ or $(128, 64)$ for CDNA3 Wave64.
- **Symptom: Compilation failure in `hipcc`:**
  - *Root Cause:* Target architecture mismatch.
  - *Correction:* Pass compiler flags explicitly: `--offload-arch=gfx942 -O3 -ffast-math`.

---

## 6. Passing Gate Checklist

- [ ] AFA HIP and AMD Triton kernels compile cleanly with `--offload-arch=gfx942`.
- [ ] Relative numerical accuracy against float64 un-tiled reference is $\le 1.0 \times 10^{-6}$.
- [ ] Head-to-head throughput benchmark executed against ROCm FlashAttention-2 on MI300X.
- [ ] Sustained HBM3 bandwidth exceeds $3.5\text{ TB/s}$.
- [ ] Assembly dump confirms zero transcendental library calls.
- [ ] `results/phase6/PASS.md` satisfies the shared PASS record contract.
