# Phase 6: Hardware-Fused Kernel Implementation & Algebraic FlashAttention (AFA on MI300X)

## 1. Objective & Research Scope
Eliminate the inter-tile synchronization barriers and transcendental online rescaling of FlashAttention. Implement, profile, and optimize **hardware-fused HIP C++ and AMD Triton kernels** specifically engineered for this server's **1x AMD Instinct MI300X (192 GB HBM3, CDNA3 `gfx942`)**:
- Implement **Algebraic FlashAttention (AFA)**: pure additive tile accumulation without running-maximum subtraction or transcendental rescaling.
- Implement and profile a high-performance **ROCm/HIP baseline FlashAttention** (using AMD Composable Kernel or FlashAttention-2 HIP port) to ensure an uncompromising, rigorously fair hardware comparison on the exact same accelerator.
- Exploit MI300X CDNA3 hardware primitives: Wave64 execution, 64 KB Local Data Share (LDS) per compute unit, and hardware $\operatorname{rsqrt}$ / matrix multiply instructions.

---

## 2. Mathematical Formulations & Hardware Execution Model

### 2.1 Pure Additive Tile Accumulation on CDNA3
For query block $\mathbf{Q}_b \in \mathbb{R}^{B_q \times d}$ and key-value blocks $\mathbf{K}_c, \mathbf{V}_c \in \mathbb{R}^{B_k \times d}$:
1. Compute raw algebraic attention scores in LDS using MFMA matrix instructions:
   $$\mathbf{S}_{bc} = \frac{\mathbf{Q}_b \mathbf{K}_c^\top}{\sqrt{d_k}}$$
2. Apply the octic algebraic kernel $\kappa_8$ in-register using CDNA3 hardware $\operatorname{rsqrt}$ and 3 squaring operations:
   $$\mathbf{P}_{bc} = \kappa_8(\mathbf{S}_{bc}) = \left(\mathbf{S}_{bc} + \sqrt{1 + \mathbf{S}_{bc}^{\odot 2}}\right)^8$$
3. Pure additive accumulation: Accumulate tile numerators $\mathbf{O}_b$ and denominators $\mathbf{D}_b$ directly into FP32 registers:
   $$\mathbf{O}_b = \sum_{c=1}^{N_{\text{tiles}}} \mathbf{P}_{bc} \mathbf{V}_c, \quad \mathbf{D}_b = \Omega + \sum_{c=1}^{N_{\text{tiles}}} \sum_{j=1}^{B_k} \mathbf{P}_{bc, \cdot j}$$
4. Final tile normalization requires zero inter-tile synchronization:
   $$\mathbf{Y}_b = \mathbf{O}_b \oslash \mathbf{D}_b$$

### 2.2 CDNA3 Architectural Alignment (MI300X)
- **Wavefront Size:** Native Wave64 (`warp_size = 64`).
- **Memory Bandwidth:** Capitalize on MI300X's $5.3\text{ TB/s}$ HBM3 bandwidth to achieve near-roofline memory saturation.
- **Register Reuse:** Keep the squaring accumulator in VGPRs (Vector General-Purpose Registers) across the 3 stages to eliminate SRAM round-trips.

---

## 3. Kernel Deliverables & Passing Gate on MI300X

The agent must deliver and benchmark two sets of kernels on this MI300X server:
1. `analysis/kernels/algebraic_attention_hip.cpp` & `analysis/kernels/algebraic_attention_triton.py` (AFA).
2. `analysis/kernels/baseline_flash_attention_hip.cpp` & ROCm FlashAttention baseline.

### Numerical & Performance Passing Criteria:

| Metric | Target on MI300X | Tolerance / Bound |
| :--- | :--- | :--- |
| **AFA Numerical Equivalence** | $\|\mathbf{Y}_{\text{AFA}} - \mathbf{Y}_{\text{exact}}\|_\infty / \|\mathbf{Y}_{\text{exact}}\|_\infty$ | $\leq 1.0 \times 10^{-6}$ |
| **Inter-Tile Rescaling FLOPs** | Transcendental $\exp(m_{\text{old}} - m_{\text{new}})$ calls in AFA | Exactly $0$ |
| **Kernel Throughput at $L=4096$** | TFLOPS on MI300X (BF16 forward) | $\geq 85\%$ of baseline FlashAttention throughput |
| **HBM Memory Bandwidth Utilization** | Sustained GB/s during tile streaming | $\geq 3.5\text{ TB/s}$ ($> 65\%$ of $5.3\text{ TB/s}$ theoretical) |
| **Zero Transcendental Audit** | ISA inspection via `llvm-objdump -d` / assembly dump | Exactly $0$ transcendental transcendentals / trigonometric opcodes |

---

## 4. Failure Modes & Self-Correction Playbook (ROCm / HIP)

- **Symptom: High VGPR pressure spills to scratch memory on MI300X:**
  *Root Cause:* Tile size $(B_q, B_k) = (128, 128)$ with unrolled squaring exceeds 256 vector registers per work-item.
  *Correction:* Tune tile dimensions to $(64, 64)$ or $(128, 64)$ for CDNA3, or serialize the squaring stages into a compact loop macro reusing 1 vector register.
- **Symptom: ROCm compilation error (`hipcc` target arch mismatch):**
  *Root Cause:* Defaulting to generic amdgpu target instead of MI300X CDNA3.
  *Correction:* Set compiler flag explicitly: `--offload-arch=gfx942` and include `-O3 -ffast-math`.

---

## 5. Passing Gate Checklist
- [ ] AFA HIP and AMD Triton kernels compile cleanly with `--offload-arch=gfx942`.
- [ ] Relative numerical accuracy against float64 un-tiled reference is $\leq 10^{-6}$.
- [ ] Head-to-head throughput benchmark executed against ROCm FlashAttention on 1x MI300X.
- [ ] Assembly dump confirms zero transcendental math library calls.
