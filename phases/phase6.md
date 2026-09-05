# Phase 6: Hardware-Fused Kernel Implementation & Algebraic FlashAttention (AFA)

## 1. Objective & Research Scope
Eliminate the inter-tile synchronization barriers and transcendental online rescaling of FlashAttention. Implement and profile **Algebraic FlashAttention (AFA)** in hardware-fused kernels (Triton / PyTorch C++/CUDA):
- Standard FlashAttention requires computing running maximums $m_{\text{tile}} = \max(\mathbf{s})$ and scaling previous tile accumulators by transcendental factors $\exp(m_{\text{old}} - m_{\text{new}})$.
- AFA replaces exponential rescaling with **purely additive tile accumulation**, enabling lock-free asynchronous Ring Attention across distributed nodes with a single global AllReduce.

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 Pure Additive Tile Accumulation
For query block $\mathbf{Q}_b \in \mathbb{R}^{B_q \times d}$ and key-value blocks $\mathbf{K}_c, \mathbf{V}_c \in \mathbb{R}^{B_k \times d}$:
1. Compute raw algebraic attention scores:
   $$\mathbf{S}_{bc} = \frac{\mathbf{Q}_b \mathbf{K}_c^\top}{\sqrt{d_k}}$$
2. Apply the octic algebraic kernel $\kappa_8$ elementwise via 3 squaring stages:
   $$\mathbf{P}_{bc} = \kappa_8(\mathbf{S}_{bc}) = \left(\mathbf{S}_{bc} + \sqrt{1 + \mathbf{S}_{bc}^{\odot 2}}\right)^8$$
3. Accumulate tile numerator and denominator additively in fast FP32 SRAM scratchpad:
   $$\mathbf{O}_b = \sum_{c=1}^{N_{\text{tiles}}} \mathbf{P}_{bc} \mathbf{V}_c, \quad \mathbf{D}_b = \Omega + \sum_{c=1}^{N_{\text{tiles}}} \sum_{j=1}^{B_k} \mathbf{P}_{bc, \cdot j}$$
4. Final tile normalization requires a single reciprocal multiply:
   $$\mathbf{Y}_b = \mathbf{O}_b \oslash \mathbf{D}_b$$

### 2.2 Theorem of Asynchronous Ring Attention Commutativity
Because tile accumulation is strictly additive ($\mathbf{O}_{\text{total}} = \sum \mathbf{O}_c$ and $\mathbf{D}_{\text{total}} = \sum \mathbf{D}_c$), distributed nodes in a Ring Attention topology can process sequence chunks in arbitrary order without waiting for neighbor normalization states.

---

## 3. Implementation & Benchmark Gate

The agent must develop and verify `analysis/algebraic_flash_attention.py` (or Triton kernel):

| Metric | Target Value | Tolerance / Bound |
| :--- | :--- | :--- |
| **Distributed Tile Equivalence** | $\|\mathbf{Y}_{\text{distributed}} - \mathbf{Y}_{\text{exact}}\|_\infty / \|\mathbf{Y}_{\text{exact}}\|_\infty$ | $\leq 1.0 \times 10^{-6}$ |
| **Inter-Tile Rescaling FLOPs** | Number of $\exp(m_{\text{prev}} - m_{\text{new}})$ operations | Exactly $0$ |
| **SRAM Scratch Memory Footprint** | Per-thread block SRAM allocation | $\leq 64 \text{ KB}$ |
| **Wall-Clock Speedup vs. Naive Attention** | At context length $L = 4096$ | $\geq 2.5\times$ speedup |

---

## 4. Failure Modes & Self-Correction Playbook

- **Symptom: SRAM overflow when computing $\kappa_8$ in FP32:**
  *Root Cause:* Storing intermediate squaring stages $\kappa_1, \kappa_2, \kappa_4$ simultaneously in registers.
  *Correction:* Reuse the same register in-place: $x \leftarrow x + \sqrt{1 + x^2}$; $x \leftarrow x^2$; $x \leftarrow x^2$; $x \leftarrow x^2$.
- **Symptom: Denominator $\mathbf{D}_b$ numerical drift across multi-node AllReduce:**
  *Root Cause:* Summing millions of small kernel outputs in FP16/BF16 leads to catastrophic cancellation.
  *Correction:* Denominator and numerator partial sums must be accumulated in FP32 scratch registers before final division.

---

## 5. Passing Gate Checklist
- [ ] AFA additive tile accumulation passes relative numerical accuracy ($\leq 10^{-6}$).
- [ ] Profiler confirms zero transcendental instructions in kernel PTX/IR.
- [ ] Distributed Ring Attention test runs successfully across 8 simulated nodes without communication barriers.
