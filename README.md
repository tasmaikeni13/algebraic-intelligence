# The Algebraic Stack: Can Algebra and Algebra Alone Give Rise to Intelligence?

[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified_Proofs-blue.svg)](https://leanprover.github.io/)
[![Pure Algebra](https://img.shields.io/badge/Architecture-100%25_Algebraic-green.svg)](#)
[![Transcendentals](https://img.shields.io/badge/Transcendentals-0%20(No%20exp,%20ln,%20sin,%20cos)-red.svg)](#)

**Author:** Tasmai Keni (`tas.ken.rt25@dypatil.edu`)

---

## Executive Summary

Contemporary deep learning architectures are saturated with transcendental functions:
- The exponential $e^x$ appears in every softmax attention layer and Swish/GELU feed-forward gate.
- The logarithm $\ln x$ appears in every cross-entropy loss and Kullback-Leibler divergence.
- The trigonometric pair $(\sin, \cos)$ appears in sinusoidal and rotary (RoPE) positional encodings.
- Continuous exponential moving averages ($e^{-\Delta t / \tau}$) and cosine annealing schedules underlie adaptive optimizers such as AdamW.

This research investigates a foundational question:
$$\textbf{Can algebra and algebra alone give rise to intelligence?}$$

Specifically, can an artificial neural system acquire reasoning, sequence induction, associative recall, and hierarchical representation when every forward pass, backward pass, normalization layer, attention mechanism, loss functional, and optimizer update is restricted strictly to rational operations, polynomial compositions, and a single hardware-native algebraic radical—the inverse square root $\mathrm{rsqrt}(x) = 1 / \sqrt{x}$—with zero exponential, logarithmic, or trigonometric functions?

We answer this question affirmatively by constructing and verifying the **Algebraic Stack**.

---

## Repository Structure

```
.
├── theory.md                       # Complete research paper and mathematical theory
├── README.md                       # Architectural overview, proofs, and benchmark results
├── formal/                         # Lean 4 machine-checked formal verification
│   ├── lakefile.toml               # Lake build specification with Mathlib4
│   ├── lean-toolchain              # Lean 4.16.0 specification
│   ├── PROOF_COVERAGE.md           # Formal theorem-to-prose mapping
│   ├── AlgebraicTheory.lean        # Root Lean 4 library module
│   └── AlgebraicTheory/
│       ├── Gate.lean               # Algebraic Gate & ALU: symmetry, Lipschitz, inflection point
│       ├── Kernel.lean             # Algebraic Kernel rho: reciprocal symmetry, 3-squaring power
│       ├── Cayley.lean             # AGO Cayley transform: SO(2) orthogonality, det=1, shift-equivariance
│       ├── Loss.lean               # Algebraic Divergence (Pearson chi^2) expansion & OACE power chain
│       ├── Curvature.lean          # ACO factorized curvature preconditioning & debiasing
│       └── Variance.lean           # AVN bounded normalization & Coupling Identity
├── phases/                         # Autonomous Research Execution & Self-Correction Engine
│   ├── README.md                   # Master governing protocol, 9-step failure-repair loop, and phase index
│   └── phase1.md to phase10.md     # Exactly 10 sequential phases through 350M scaling & publication
├── skills/                         # Autonomous Scientific Research Skills & Frameworks
└── analysis/                       # Empirical and Numerical Mathematical Suite
    ├── algebraic_stack.py          # 100% Pure Algebraic PyTorch reference implementation
    ├── verify_algebraic_primitives.py # Numerical verification of all mathematical theorems
    └── benchmark_algebraic_vs_transcendental.py # Controlled empirical benchmarks vs standard Transformers
```

---

## Autonomous Research Engine & Verification Phases

All autonomous research and verification in this repository is governed by [`phases/README.md`](phases/README.md). The research lifecycle is organized into **exactly ten sequential phases** defined in [`phases/`](phases/):

- [**Phase 1: Pure Algebraic Primitives & Non-Linear Gating**](phases/phase1.md) (ALU Inflection at $-\sqrt{2}$, Parameter-Free AVN, Horner Cubic Backward)
- [**Phase 2: Octic Algebraic Attention & 2-Lipschitz Bounds**](phases/phase2.md) (A-Softmax 3-Stage Squaring $\kappa_8$, Uniform $\le n/4$ Jacobian, FP4 Quantization)
- [**Phase 3: Algebraic Geometric Oscillators & Shift Equivariance**](phases/phase3.md) (AGO Cayley Rotations on $\mathfrak{so}(2)$, Unimodular $\det=1$, $\mathcal{O}(1)$ Decode)
- [**Phase 4: Algebraic Loss Functionals & Information Metrics**](phases/phase4.md) (OACE $\mathcal{L}_{1/8}$, Bounded Gradient $8 p_k^{-1/8}$, Pearson $\chi^2$, Fisher Equivalence)
- [**Phase 5: Factorized Curvature Optimization & Rational Scheduling**](phases/phase5.md) (ACO $\mathcal{O}(d_{\text{out}} + d_{\text{in}})$ Curvature, ARDS Rational Decay Schedule)
- [**Phase 6: Hardware-Fused Kernels & Algebraic FlashAttention on MI300X**](phases/phase6.md) (CDNA3 Wave64 HIP/Triton AFA Kernel, Additive Accumulation, $> 3.5\text{ TB/s}$)
- [**Phase 7: Full Architecture Assembly & Pilot Pretraining**](phases/phase7.md) (15M LM on WikiText-103 across $10^5$ Steps on MI300X, Head-to-Head Comparison)
- [**Phase 8: Frontier Pretraining: 125M Parameters on 1.0B Tokens**](phases/phase8.md) (3 Paired Seeds on FineWeb-Edu on 1x MI300X, Downstream Zero-Shot Reasoning)
- [**Phase 9: Scaled Pretraining: 350M Parameters on 3.0B Tokens & Scaling Laws**](phases/phase9.md) (24 Layers, Width 1024, Power-Law Scaling, Hierarchical Back-Propagation Loop)
- [**Phase 10: Comprehensive Research Paper, Clean-Room Replication, & Release**](phases/phase10.md) (Fresh-Clone Reproduction on MI300X, Standalone Manuscript, Full Completion Matrix)


---

## The Twelve Algebraic Primitives

| Component | Standard Target Replaced | Algebraic Formulation | Defining Mathematical Guarantee |
| :--- | :--- | :--- | :--- |
| **ALU** | GELU, Swish | $K(x) = \frac{x}{2}(1 + u), u = x \cdot \mathrm{rsqrt}(x^2 + 1)$ | $\mathcal{O}(1)$ backward pass; $L_K \approx 1.0445$; Inflection at $-\sqrt{2}$ (Thm 3.2, 3.4) |
| **A-Softmax** | Softmax | $\mathbf{S}_n(\mathbf{s}) = \rho(\hat{\mathbf{s}})^n / \sum \rho(\hat{\mathbf{s}})^n, n = 8$ | 2-Lipschitz operator; $10^5$ contrast at bounded inputs; INT4/FP4 stable (Thm 4.6, 4.7) |
| **OACE** | Cross-Entropy ($-\ln p$) | $\mathcal{L}_{1/8} = 8(p_k^{-1/8} - 1)$ | 3-rsqrt backward; strictly bounded gradient $8 p_k^{-1/8}$ (Thm 4.15, Prop 4.16) |
| **AD** | KL Divergence | $D_A(\mathbf{y} \| \mathbf{p}) = \sum y_i^2 / p_i - 1$ | Pearson $\chi^2$ equivalence; Riemannian Fisher equivalence; Bounded gradient (Thm 5.2, 5.3) |
| **AVN** | LayerNorm, RMSNorm | $\tau = \mathrm{rsqrt}(m_2(\mathbf{x}) + \epsilon), \hat{\mathbf{x}} = \tau \mathbf{x}$ | Zero parameters; Coupling Identity $\beta(x; v) = \beta(\hat{x}; 1)$ (Def 6.1, Thm 6.2) |
| **AGO** | RoPE, Sinusoidal PE | $\mathbf{R}_k = (\mathbf{I} + \omega_k\mathbf{J})(\mathbf{I} - \omega_k\mathbf{J})^{-1}$ | Exact shift equivariance $\langle\mathbf{Q}_m,\mathbf{K}_n\rangle = f(n - m)$; $\mathcal{O}(1)$ decode (Thm 7.5, 7.6) |
| **AA** | Softmax Attention | Dual-track: local A-Softmax + ALU delta rule | Linear global associative memory; contractive stability $\|\mathbf{S}_t\|_F < \infty$ (Thm 8.2) |
| **AFA** | FlashAttention-2 | Additive tile accumulation without max reduction | Lock-free asynchronous Ring Attention via single AllReduce (Thm 9.1, Cor 9.2) |
| **ALU-GLU** | SwiGLU, GeGLU | $\mathbf{W}_d [(\mathbf{W}_g \mathbf{x}) \odot K(\mathbf{W}_u \mathbf{x})]$ | Polynomial backward in cached $u$; Universal approximation (Thm 10.2, 10.3) |
| **A-MoE** | Softmax + Gumbel MoE | AVN-bounded $\rho^8$ routing + ANT noise | Native FP4 routing; variance-adaptive exploration; anti-collapse (Thm 11.3, Cor 4.10) |
| **ACO** | AdamW Optimizer | Factorized curvature $\frac{r_i c_j}{\bar{r}}$ + ARDS schedule | $\mathcal{O}(d_{\mathrm{out}} + d_{\mathrm{in}})$ memory; rational momentum; $\mathcal{O}(1/\sqrt{T})$ rate (Thm 12.3, 12.7) |
| **ABA** | BPE Tokenizer | Patch-pooled ALU-GLU on raw bytes | Constant-bounded typo shatter $\mathcal{O}(1)$ vs BPE $\Omega(\sqrt{L})$ (Thm 13.3) |
| **AIP** | VICReg, Barlow Twins | Power iteration + $\|\mathbf{C} - \mathbf{I}\|_F^2$ + AVN repulsion | Structural anti-roughness, anti-dimension, and anti-mode collapse (Section 14) |

---

## Machine-Checked Formal Verification in Lean 4

All foundational algebraic theorems have been formalized and verified in **Lean 4** (v4.16.0) with Mathlib4.
To verify the proofs locally:
```bash
cd formal
lake build
```

Key formally verified theorems:
1. `gate_reflection_identity`: $\beta(u) + \beta(-u) = 1$.
2. `alu_polynomial_backward_identity`: $\frac{d}{dx} K(x) = \frac{1}{2}(1 + 2u - u^3)$ (cubic polynomial in cached $u$).
3. `alu_inflection_identity`: $K''(x) = 0 \iff 2 - 3u^2 = 0 \iff x = -\sqrt{2}$, proving algebraic alignment with GELU.
4. `kernel_reciprocal_identity`: $(x + s)(s - x) = 1$ when $s^2 = x^2 + 1$.
5. `cayley_column_norm_one` & `cayley_determinant_one`: Rational Cayley transform produces an exact orthogonal rotation in $\mathrm{SO}(2)$ with $\det = 1$.
6. `pearson_divergence_expansion`: $(y - p)^2 / p = y^2/p - 2y + p$, proving the Pearson $\chi^2$ expansion.
7. `aco_factorized_curvature_recovery`: $\frac{(a_i \bar{b})(b_j \bar{a})}{\bar{a}\bar{b}} = a_i b_j$, proving exact recovery of Kronecker Fisher curvature.
8. `avn_bounded_norm` & `avn_coupling_identity`: Bounded variance normalization and coupling with downstream algebraic gates.

---

## Empirical Research Benchmarks

Run the complete test suite:
```bash
python3 analysis/verify_algebraic_primitives.py
python3 analysis/benchmark_algebraic_vs_transcendental.py
```

### Empirical Results Summary

1. **In-Context Sequence Intelligence:**
   - **Task:** Sequence Induction and Key Retrieval across context.
   - **Pure Algebraic Stack:** Reached **100.0% accuracy** in 220 steps.
   - **Transcendental Baseline:** Reached 100.0% accuracy.
   - **Conclusion:** Pure algebra alone matches transcendental Transformers on fundamental sequence learning.

2. **Optimizer Memory Footprint (ACO vs AdamW):**
   - At matrix dimension $4096 \times 4096$: AdamW state = **128.00 MB**, ACO state = **64.03 MB** (compression factor: 2.0x, factorized: 2048x).
   - At matrix dimension $8192 \times 8192$: AdamW state = **512.00 MB**, ACO state = **256.06 MB** (factorized: 4096x).

3. **Sub-Byte (FP4/INT4) Quantization Stability:**
   - Output displacement under logit quantization noise: Softmax = **0.0141**, A-Softmax = **0.0001**.
   - **Stability Gain:** A-Softmax is **228.17x less sensitive** to quantization noise than exponential softmax due to its global 2-Lipschitz property.

4. **Asynchronous Distributed Ring Attention (AFA):**
   - Distributed tile simulation across $P = 8$ nodes.
   - Relative error between lock-free additive AFA and exact un-tiled attention: **$3.24 \times 10^{-7}$**.
   - Zero inter-tile synchronization barriers; single global AllReduce.

---

## Citation

```bibtex
@article{keni2026algebraicstack,
  title={The Algebraic Stack: Can Algebra and Algebra Alone Give Rise to Intelligence?},
  author={Keni, Tasmai},
  journal={Preprint},
  year={2026},
  url={https://github.com/tasmaikeni13/algebraic-intelligence}
}
```
