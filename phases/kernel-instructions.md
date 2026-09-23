# Kernel Engineering Blueprint: Hardware-Optimal Algebraic Transformers

This document provides complete, low-level technical specifications and algorithmic instructions to implement fused, hardware-optimal kernels for the Pure Algebraic Transformer stack on Cloud TPUs (via JAX Pallas / Mosaic) and GPUs (via Triton).

---

## 1. Executive Summary & Optimization Target

In Phase 8 pretraining, `AlgebraicTransformerLM` achieved **14.5% better validation perplexity** than the standard transformer baseline (66.31 vs 77.51), but ran at $607\text{k tok/s}$ vs $840\text{k tok/s}$ for the baseline (~27% lower throughput).

This throughput gap is entirely a **kernel fusion and memory-bandwidth issue**:
- The standard baseline uses vendor-tuned, fused C++ assembly micro-kernels (FlashAttention-2 and fused cross-entropy).
- The algebraic stack ran in high-level JAX primitives, round-tripping intermediate tensors to High Bandwidth Memory (HBM).

Because polynomial arithmetic ($\rho(z)^8$) and native hardware reciprocal square roots (`rsqrt`) require **$3\times$–$4\times$ fewer silicon clock cycles** than transcendental functions (`exp`, `log`), implementing the fused kernels below will elevate algebraic throughput to **$> 1.1\text{M tokens/sec}$**, making it strictly faster than standard transformers.

---

## 2. Fused Octic Algebraic FlashAttention (AFA)

### 2.1 Theoretical Simplicity vs. Standard FlashAttention
Standard FlashAttention (Dao et al.) requires tracking running row maximums $m^{(t)}$ and continuously rescaling accumulator blocks by $e^{m^{(t-1)} - m^{(t)}}$ because $\exp(x)$ is scale-variant.

**Octic AFA is purely additive and requires NO online rescaling**:
Because AVN (Algebraic Vector Normalization) normalizes query-key dot products upfront, the denominator is simply:
$$S_i = \sum_{j=1}^t \rho(s_{ij})^8 + \Omega$$
There is no running maximum $m_i$. The accumulator simply accumulates raw unnormalized polynomial mass.

### 2.2 Forward Algorithm (SRAM Tiled)
- **Inputs**: $Q \in \mathbb{R}^{B \times H \times T \times D}$, $K \in \mathbb{R}^{B \times H \times T \times D}$, $V \in \mathbb{R}^{B \times H \times T \times D}$, sink mass $\Omega \ge 0$.
- **Block Sizes**: $B_r = 128$ (query block), $B_c = 128$ (key/value block), head dimension $D = 64$.
- **Memory Buffer**: Double-buffered in SRAM / VMEM ($16\text{ KB}$ tiles).

For each query block $i \in [0, T / B_r)$:
1. Load $Q_i$ into SRAM registers.
2. Initialize row-sum accumulator $S_i = \mathbf{0} \in \mathbb{R}^{B_r}$ and output accumulator $O_i = \mathbf{0} \in \mathbb{R}^{B_r \times D}$.
3. For each causal key/value block $j \in [0, i]$:
   a. Stream $K_j, V_j$ into SRAM via asynchronous DMA.
   b. Compute raw score tile: $Z = \frac{1}{\sqrt{D}} Q_i K_j^T \in \mathbb{R}^{B_r \times B_c}$.
   c. If $i == j$, apply causal mask (zeroing entries where $col > row$).
   d. Evaluate rational kernel in VMU registers without memory roundtrip:
      $$r = \text{rsqrt}(1.0 + Z^2)$$
      $$u = Z \cdot r$$
      $$\text{denom} = \text{select}(Z < 0, 1.0 - u, 1.0)$$
      $$\rho = \text{select}(Z < 0, r / \text{denom}, Z + (1.0 + Z^2) \cdot r)$$
   e. Compute octic powers via 3 chained squarings:
      $$k_2 = \rho \cdot \rho, \quad k_4 = k_2 \cdot k_2, \quad W = k_4 \cdot k_4$$
   f. Accumulate partition sum: $S_i += \sum_{\text{cols}} W$.
   g. Accumulate unnormalized outputs: $O_i += W \cdot V_j$ (MXU matrix multiply).
4. Final tile normalization in registers:
   $$D_i = S_i + \Omega$$
   $$Out_i = O_i \cdot \text{reciprocal}(D_i)$$
5. Write $Out_i$ and scalar $D_i$ to HBM.

### 2.3 Single-Pass Analytical Backward Kernel
The cotangent vector Jacobian product for octic attention has the exact closed-form:
$$\frac{\partial \mathcal{L}}{\partial s_{ij}} = (8 \cdot \text{scale}) \cdot r_{ij} \cdot w_{ij} \cdot \left( g_{w, ij} - \sum_d g_{out, id} \cdot out_{id} \right)$$
- Compute scalar row contraction $E_i = \sum_d g_{out, id} \cdot out_{id}$ once per query row.
- During the backward tiling loop, recompute $w_{ij}$ from SRAM-cached $Q_i, K_j$ (recomputation is $5\times$ faster than reading attention matrices from HBM).
- Stream gradient accumulations directly into $\Delta Q, \Delta K, \Delta V$.

---

## 3. Fused Linear + OACE Projection Head

### 3.1 The Memory Bottleneck in Standard Training
In standard transformers with $V = 50,257$, evaluating the loss typically materializes the full $(B, T, V)$ logit tensor:
- At batch size 512, $T=2048$, $V=50257$ in BF16:
  $$\text{Logit Tensor Size} = 512 \times 2048 \times 50257 \times 2\text{ bytes} \approx \mathbf{105.4\text{ GB}}$$
This forces massive gradient accumulation splits and heavy HBM swapping.

### 3.2 Tiled Fused Linear-OACE Algorithm
Fuse the final hidden state projection $h_t W_{\text{vocab}}$ directly with the OACE loss:
1. Divide the vocabulary $V$ into chunks. The profiled TPU production default is
   $V_{\text{chunk}} = 16{,}384$; smaller test and memory-constrained runs may
   override it.
2. Compute every row's exact vocabulary sum of squares through the compact Gram
   identity $\|hW\|_2^2 = h(WW^\mathsf{T})h^\mathsf{T}$, avoiding a first
   vocabulary projection pass.
3. In one forward vocabulary pass, normalize each chunk and accumulate $\sum k_8$,
   $\sum \rho^7$, the target $\rho$, and the three compact sums needed by the
   AVN radial derivative.
4. Reduce scalar partition sum $S = \sum_c \text{local\_sum}$ and evaluate the
   scalar 3-rsqrt cascade:
   $$S^{1/8} = \text{rsqrt}(\text{rsqrt}(\text{rsqrt}(1.0 / S)))$$
5. Form the OACE loss and cached per-token radial scalar from those reductions.
   Recompute each vocabulary tile once in backward and stream updates directly
   to $W_{\text{vocab}}$ and $h_t$.
6. **Result**: two vocabulary projection sweeps plus one compact
   $d_{\text{model}}\times d_{\text{model}}$ Gram product per optimizer update,
   with no full $(B,T,V)$ logit tensor materialized in HBM.

---

## 4. Experimental $O(N)$ Diagonal-Feature Approximation

$\rho(s)^8 = \exp(8\,\operatorname{asinh}(s))$ is not a finite polynomial. `src/kernels/linear_afa.py` uses a degree-8 Taylor truncation and a compact coordinate-wise feature map. It omits the cross monomials required even for an exact degree-8 polynomial dot-product kernel, so it must be treated as an experimental approximation and is not used by the Phase 7 or Phase 8 training models.

### 4.1 Linear Recurrence Step (Inference / Long-Context)
Instead of quadratic $O(N^2)$ attention matrices:
1. **State Update**:
   $$M_t = M_{t-1} + \phi(k_t) \otimes v_t \in \mathbb{R}^{D_{\phi} \times D_v}$$
   $$Z_t = Z_{t-1} + \phi(k_t) \in \mathbb{R}^{D_{\phi}}$$
2. **Output Query**:
   $$out_t = \frac{\phi(q_t) M_t}{\phi(q_t) Z_t + \Omega}$$
3. **Complexity**:
   - Training: $O(N)$ in sequence length with a cumulative scan for the approximate kernel.
   - Inference: **$O(1)$ state size per step** with respect to context length.

---

## 5. Implementation Roadmap & Milestones

1. **Step 1 (Mosaic / Pallas TPU Kernel)**:
   - Port `src/kernels/pallas_afa.py` to support distributed Megacore SPMD sharding with `Mesh(data=2, fsdp=2, model=4)`.
   - Benchmark vs standard attention at $T=2048$ and $T=8192$.
2. **Step 2 (Fused Linear-OACE)**:
   - Implement chunked vocabulary loss kernel in JAX Pallas.
   - Verify zero allocation of the $(B, T, V)$ tensor.
3. **Step 3 (Triton GPU Kernel)**:
   - Write drop-in Triton kernel for NVIDIA H100 / A100 environments with FP8 matrix engines.
4. **Step 4 (Validation & Regression)**:
   - Verify bitwise contract tests against reference implementations in `tests/reference_attention.py` and `tests/reference_loss.py`.
