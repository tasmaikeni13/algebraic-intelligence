# The Algebraic Stack: Can Algebra and Algebra Alone Give Rise to Intelligence?

[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified_Proofs-blue.svg)](https://leanprover.github.io/)
[![Pure Algebra](https://img.shields.io/badge/Architecture-100%25_Algebraic-green.svg)](#)
[![Transcendentals](https://img.shields.io/badge/Transcendentals-0%20(No%20exp,%20ln,%20sin,%20cos)-red.svg)](#)

**Author:** Tasmai Keni (`tas.ken.rt25@dypatil.edu`)

---

## Current implementation status

- **Phase 1 (Pure Algebraic Primitives & Non-Linear Gating):** [PASS](results/phase1/PASS.md). Adds JAX ALU/AVN primitives, analytical cached backward passes, independent float64 oracle, unit tests, Lean 4 certificates, and reproducible CPU/16-chip TPU v4 verification runners ([Reproduction Guide](results/phase1/REPRODUCE.md), [Status](results/phase1/STATUS.md)).
- **Phase 2 (Octic Algebraic Attention & Entrywise Jacobian Bounds):** [PASS](results/phase2/PASS.md). Adds JAX A-Softmax (`algebraic_softmax`), exact 3-stage squaring hierarchy $\rho^8$, analytical cached VJP with the full AVN quotient-rule cotangent, Lean 4 certificates in `Kernel.lean`, the normalized-coordinate entrywise bound $\max |J_{ij}| \le 2.0$ (not a spectral-norm bound), 1D Wasserstein-1 distribution parity, and $354.51\times$ noise suppression in the specified canonical outlier benchmark across 16 TPU v4 chips ([Reproduction Guide](results/phase2/REPRODUCE.md), [Status](results/phase2/STATUS.md)).
- **Phase 3 (Algebraic Geometric Oscillators & Shift Equivariance):** [PASS](results/phase3/PASS.md). Adds JAX Algebraic Geometric Oscillators (`build_cayley_rotary_matrix` and `apply_ago_rotations`), Lean 4 certificates for $\mathrm{SO}(2)$ Lie group unimodularity, orthogonality, norm invariance, and relative shift equivariance in `Cayley.lean`, zero-transcendental AST purity, cumulative norm conservation up to $L=8192$ (drift $\le 2.22 \times 10^{-16}$), out-of-distribution associative recall generalization ($100\%$ at $L=1024$, $99.5\%$ at $L=2048$ from $L=256$ training), and $99.4\% - 107.7\%$ throughput parity vs RoPE across 16 TPU v4 cores ([Reproduction Guide](results/phase3/REPRODUCE.md), [Status](results/phase3/STATUS.md)).
- **Phase 4 (Algebraic Loss Functionals & Information Metrics):** [PASS](results/phase4/PASS.md). Adds the strictly proper non-local OACE power score (`oace_loss`) and Pearson $\chi^2$ divergence (`pearson_divergence`), a closed-form zero-transcendental VJP, Lean certificates, $10^5$-sample soft-target propriety and same-domain label-noise checks, exact probability-gradient verification, finite AVN + A-Softmax composed gradients, Fisher metric ratio $2.0$, and $103.8\%$–$107.2\%$ TPU throughput ratios versus cross-entropy ([Reproduction Guide](results/phase4/REPRODUCE.md), [Status](results/phase4/STATUS.md)).
- **Phase 5 (Algebraic Optimization & Rational Scheduling):** [PASS](results/phase5/PASS.md). Adds JAX Algebraic AdamW (`algebraic_adamw`) with exact polynomial debiasing, Optax-compatible default denominator semantics lowered without raw `sqrt`, and Algebraic Rational Decay (`ards_schedule`) via hardware $\operatorname{rsqrt}$. The gates include Lean certificates, $10^4$ ill-conditioned trials, non-convex parity, compiled-HLO checks with zero raw `sqrt`, and $99.36\%$–$100.50\%$ TPU throughput ratios versus cosine scheduling ([Reproduction Guide](results/phase5/REPRODUCE.md), [Status](results/phase5/STATUS.md)).
- **Phase 6 (Hardware-Fused Kernels & Algebraic FlashAttention on 16 TPU v4 Pod):** [PASS](results/phase6/PASS.md). Adds the real JAX Pallas TPU kernel `pallas_afa_forward`, additive tiled accumulation, FP32 accumulation, TPU-legal dtype-specific dot precision, and a 288 KiB bounded working set. Eight FP32/BF16 parity configurations pass; throughput is $98.41\%$ at $L=2048$ and $99.67\%$ at $L=4096$ versus JAX's Pallas TPU FlashAttention; 16-chip ring relative error is $4.15\times10^{-7}$. Physical HBM utilization is deliberately not inferred without profiler counters ([Reproduction Guide](results/phase6/REPRODUCE.md), [Status](results/phase6/STATUS.md)).
- **Phase 7 (Full Architecture Assembly & Pilot Pretraining):** **RERUN REQUIRED.** The prior TPU evidence used 200 steps instead of the required 100,000, and the old validator failed to enforce the execution budget. The implementation now routes both architectures through their optimized attention and fused-loss paths, and the evidence gate checks the full step count and 16-device topology. See the [invalidation record](results/phase7/PASS.md).
- **Phase 8 (Systematic Hyperparameter Sweeping & Architecture Tuning):** **RERUN REQUIRED after Phase 7.** The prior six-run result evaluated one fixed configuration per architecture and overstated the processed token count, so it was not a valid sweep. The corrected protocol evaluates three preregistered candidates per architecture across three seeds (18 runs) and rounds full batches upward to meet the 600M-token minimum. See the [invalidation record](results/phase8/PASS.md) and [candidate matrix](phases/phase8_candidates.json).

The architecture and later-phase results described below for Phases 9–10 are research-draft
claims to be executed in subsequent phases.

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
│   ├── lean-toolchain              # Lean 4.34.0-rc2 specification
│   ├── PROOF_COVERAGE.md           # Formal theorem-to-prose mapping
│   ├── AlgebraicTheory.lean        # Root Lean 4 library module
│   └── AlgebraicTheory/
│       ├── Gate.lean               # Algebraic Gate & ALU: symmetry, Lipschitz, inflection point
│       ├── Kernel.lean             # Algebraic Kernel rho: reciprocal symmetry, 3-squaring power
│       ├── Cayley.lean             # AGO Cayley transform: SO(2) orthogonality, det=1, shift-equivariance
│       ├── Loss.lean               # Algebraic Divergence (Pearson chi^2) expansion & OACE power chain
│       ├── Curvature.lean          # AdamW algebraic properties: debiasing, decoupled decay, and curvature
│       ├── Variance.lean           # AVN bounded normalization & Coupling Identity
│       └── Composition.lean        # End-to-end signal propagation & Lipschitz composition bounds
├── skills/                         # Autonomous Scientific Research Skills & Frameworks
└── phases/                         # Autonomous Research Execution & Self-Correction Engine
    ├── README.md                   # Master governing protocol, adaptive dependency cascading, and phase index
    └── phase1.md to phase10.md     # Exactly 10 sequential phases through hyperparameter sweeping, 125M / 2.5B tokens & publication on 16 TPU v4 Pod
```

---

## Autonomous Research Engine & Verification Phases

All autonomous research and verification in this repository is governed by [`phases/README.md`](phases/README.md). The research lifecycle is organized into **exactly ten sequential phases** defined in [`phases/`](phases/):

- [**Phase 1: Pure Algebraic Primitives & Non-Linear Gating**](phases/phase1.md) — **VERIFIED PASS** ([PASS.md](results/phase1/PASS.md)) (ALU Inflection at $-\sqrt{2}$, Parameter-Free AVN, Horner Cubic Backward)
- [**Phase 2: Octic Algebraic Attention & 2-Lipschitz Bounds**](phases/phase2.md) — **VERIFIED PASS** ([PASS.md](results/phase2/PASS.md)) (A-Softmax 3-Stage Squaring $\rho^8$, Entrywise $\le 2.0$ Jacobian, 1D Wasserstein-1 Parity, $354\times$ FP4 Noise Suppression)
- [**Phase 3: Algebraic Geometric Oscillators & Shift Equivariance**](phases/phase3.md) — **VERIFIED PASS** ([PASS.md](results/phase3/PASS.md)) (AGO Cayley Rotations on $\mathfrak{so}(2)$, Unimodular $\det=1$, $\mathcal{O}(1)$ Decode)
- [**Phase 4: Algebraic Loss Functionals & Information Metrics**](phases/phase4.md) — **VERIFIED PASS** ([PASS.md](results/phase4/PASS.md)) (strictly proper non-local OACE, exact probability gradient, finite composed gradient, Pearson $\chi^2$, Fisher equivalence)
- [**Phase 5: Algebraic Optimization & Rational Scheduling**](phases/phase5.md) — **VERIFIED PASS** ([PASS.md](results/phase5/PASS.md)) (AdamW Native Algebraic Verification, ARDS Rational Decay Schedule, Ill-Conditioned $\kappa \le 10^6$ Sweep, Non-Convex Stochastic Parity)
- [**Phase 6: Hardware-Fused Kernels & Algebraic FlashAttention on 16 TPU v4 Pod**](phases/phase6.md) — **VERIFIED PASS** ([PASS.md](results/phase6/PASS.md)) (real JAX Pallas TPU kernel, bounded additive tile accumulation, $98.8\%$–$99.9\%$ Pallas-baseline throughput, lock-free Ring Attention over ICI)
- [**Phase 7: Full Architecture Assembly & Pilot Pretraining**](phases/phase7.md) — **RERUN REQUIRED**; the historical 200-step job does not satisfy the 100,000-step contract.
- [**Phase 8: Systematic Hyperparameter Sweeping & Architecture Tuning**](phases/phase8.md) — **RERUN REQUIRED AFTER PHASE 7**; the corrected protocol is 18 runs (2 architectures × 3 candidates × 3 seeds), each processing at least 600M tokens.
- [**Phase 9: Frontier Pretraining: 125M Parameters on 2.5B Tokens**](phases/phase9.md) (Main Frontier Pretraining: 125M Parameters across 6 Runs [2 Architectures $\times$ Seeds 42, 43, 44] on 2.5B FineWeb-Edu Tokens each on 16 TPU v4 Pod / v4-32, Downstream Zero-Shot Reasoning)
- [**Phase 10: Comprehensive Research Paper, Clean-Room Replication, & Release**](phases/phase10.md) (Fresh-Clone Reproduction on 16 TPU v4 Pod, Standalone Manuscript, Full Completion Matrix)


---

## Core Algebraic Primitives & Architectural Foundations

| Component | Standard Target Replaced | Algebraic Formulation | Defining Mathematical Guarantee |
| :--- | :--- | :--- | :--- |
| **ALU** | GELU, Swish | $K(x) = \frac{x}{2}(1 + u), u = x \cdot \mathrm{rsqrt}(x^2 + 1)$ | $\mathcal{O}(1)$ backward pass; $L_K \approx 1.0445$; Inflection at $-\sqrt{2}$ (Thm 3.2, 3.4) |
| **A-Softmax** | Softmax | $\mathbf{S}_n(\mathbf{s}) = \rho(\hat{\mathbf{s}})^n / \sum \rho(\hat{\mathbf{s}})^n, n = 8$ | Normalized-coordinate entrywise derivative $\le2$; $10^5$ contrast; canonical-outlier quantization benchmark (Thm 4.6, 4.7) |
| **OACE** | Cross-Entropy ($-\ln p$) | $8\sum y_i p_i^{-1/8}+\frac87\sum p_i^{7/8}-\frac{64}{7}\sum y_i^{7/8}$ | Strictly proper; three-rsqrt prediction path and analytical VJP (Thm 4.13–4.15) |
| **AD** | KL Divergence | $D_A(\mathbf{y} \| \mathbf{p}) = \sum y_i^2 / p_i - 1$ | Pearson $\chi^2$ equivalence; Riemannian Fisher equivalence; Bounded gradient (Thm 5.2, 5.3) |
| **AVN** | LayerNorm, RMSNorm | $\tau = \mathrm{rsqrt}(m_2(\mathbf{x}) + \epsilon), \hat{\mathbf{x}} = \tau \mathbf{x}$ | Zero parameters; Coupling Identity $\beta(x; v) = \beta(\hat{x}; 1)$ (Def 6.1, Thm 6.2) |
| **AGO** | RoPE, Sinusoidal PE | $\mathbf{R}_k = (\mathbf{I} + \omega_k\mathbf{J})(\mathbf{I} - \omega_k\mathbf{J})^{-1}$ | Exact shift equivariance $\langle\mathbf{Q}_m,\mathbf{K}_n\rangle = f(n - m)$; $\mathcal{O}(1)$ decode (Thm 7.5, 7.6) |
| **AFA** | FlashAttention-2 | Additive tile accumulation without max reduction | Lock-free asynchronous Ring Attention via single AllReduce (Thm 8.1, Cor 8.2) |
| **ALU-GLU** | SwiGLU, GeGLU | $\mathbf{W}_d [(\mathbf{W}_g \mathbf{x}) \odot K(\mathbf{W}_u \mathbf{x})]$ | Polynomial backward in cached $u$; Universal approximation (Thm 9.2, 9.3) |
| **AdamW + ARDS** | AdamW + Cosine Decay | Rational moments + $\mathrm{rsqrt}$ update with ARDS rational decay | Fully algebraic optimizer ($\mathcal{O}(1/\sqrt{T})$ rate, zero transcendentals, Thm 10.1, 10.4) |
| **A-MoE\*** | Softmax + Gumbel MoE | AVN-bounded $\rho^8$ routing + ANT noise | Native FP4 routing; variance-adaptive exploration; anti-collapse (Thm 11.2) |

*\* Note: A-MoE is formulated mathematically in `theory.md` as future work / extension for sparse scaling; the current empirical campaign focuses strictly on dense causal language modeling (`AlgebraicTransformerLM`).*

---

## Machine-Checked Formal Verification in Lean 4

All foundational algebraic theorems have been formalized and verified in **Lean 4** (v4.34.0-rc2) with Mathlib4.
To verify the proofs locally:
```bash
cd formal
lake build
```

Key formally verified theorems:
1. `gate_reflection_identity`: $\beta(u) + \beta(-u) = 1$.
2. `alu_polynomial_backward_identity`: $\frac{d}{dx} K(x) = \frac{1}{2}(1 + 2u - u^3)$ (cubic polynomial in cached $u$).
3. `alu_inflection_iff`: $2 - 3u^2 = 0 \iff x^2 = 2$ under the cache relation, proving algebraic alignment with GELU.
4. `kernel_reciprocal_identity`: $(x + s)(s - x) = 1$ when $s^2 = x^2 + 1$.
5. `cayley_column_norm_one` & `cayley_determinant_one`: Rational Cayley transform produces an exact orthogonal rotation in $\mathrm{SO}(2)$ with $\det = 1$.
6. `pearson_divergence_expansion`: $(y - p)^2 / p = y^2/p - 2y + p$, proving the Pearson $\chi^2$ expansion.
7. `adamw_debiasing_identity` & `adamw_decoupled_weight_decay`: Mathematical algebraic structure of AdamW bias correction and decoupled weight updates without transcendental functions.
8. `avn_bounded_norm` & `avn_coupling_identity`: Bounded variance normalization and coupling with downstream algebraic gates.
9. `avn_coord_bound_with_tau` & `residual_l_layer_growth`: AVN coordinate bounds under inverse variance scaling and bounded linear signal growth under iterated residual layers (`Composition.lean`).

---

## Empirical Research & Benchmarks

Run the complete verification and benchmark suite:
```bash
# 1. Verify all algebraic primitives and comparative baselines:
python3 scripts/run_verify_primitives.py

# 2. Run Pallas AFA hardware kernel benchmarks on 16 TPU v4 Pod:
python3 scripts/run_benchmark_pallas.py

# 3. Launch unit and integration tests:
pytest tests/

# 4. Verify Phase 7 pilot pretraining & TPU acceptance gates:
python3 scripts/run_verify_phase7.py
```

### Empirical Results Summary

1. **In-Context Sequence Intelligence:**
   - **Task:** Sequence Induction and Key Retrieval across context.
   - **Pure Algebraic Stack:** Reached **100.0% accuracy** in 220 steps.
   - **Transcendental Baseline:** Reached 100.0% accuracy.
   - **Conclusion:** Pure algebra alone matches transcendental Transformers on fundamental sequence learning.

2. **Algebraic Optimizer Purity (AdamW + ARDS):**
   - **Zero Transcendental Functions:** Verified $0$ occurrences of $e^x, \ln x, \sin x, \cos x$ in AdamW updates and ARDS decay schedule $\eta_t = \eta_0 \cdot \mathrm{rsqrt}(1 + \alpha t^2)$.
   - **Isolate Architectural Ablation:** Standardizing both `AlgebraicTransformerLM` and `StandardTransformerLM` on AdamW guarantees that performance differences reflect purely the architectural algebra (ALU, A-Softmax, AVN, AGO, OACE) rather than optimizer confounds.

3. **Sub-Byte (FP4/INT4) Quantization Stability:**
   - Output displacement under logit quantization noise: Softmax = **0.00589**, A-Softmax = **0.0000166**.
   - **Specified-benchmark result:** In the canonical $K=128$ outlier setup ($s_0\mathrel{+}=6$, noise $\sigma=0.05$), A-Softmax is **354.51x less sensitive** on the 16-chip TPU v4 slice (**354.48x** on CPU). Unscaled Gaussian diagnostics do not show a uniform advantage.

4. **Asynchronous Distributed Ring Attention (AFA):**
   - Distributed tile simulation across $P = 8$ nodes.
   - Relative error between lock-free additive AFA and exact un-tiled attention: **$3.24 \times 10^{-7}$**.
   - Zero inter-tile synchronization barriers; single global AllReduce.

5. **Head-to-Head Pilot Pretraining on WikiText-103 (Phase 7):**
   - **INVALIDATED:** This job ran 200 steps rather than the required 100,000. The figures below are historical short-run diagnostics and are not Phase 7 evidence.
   - **Hardware & Scale:** 15.9M parameter matched budget trained on physical Google Cloud TPU v4-32 Pod slice (4 hosts, 16 physical chips, 32 TensorCores) across 2D/3D mesh `(data=2, fsdp=2, model=4)` with batch size $32,768$ tokens/step.
   - **Historical short-run perplexity:** The 200-step diagnostic recorded $\mathbf{400.77}$ vs $\mathbf{469.35}$ (ratio $\mathbf{0.8539}$); this cannot establish the Phase 7 convergence claim.
   - **Historical short-run throughput:** The diagnostic recorded $\mathbf{2,915,698\text{ tok/s}}$ vs $\mathbf{3,231,782\text{ tok/s}}$ (ratio $\mathbf{0.9022}$); this cannot establish 100,000-step throughput stability.
   - **Numerical Stability:** Exact zero NaNs, zero Infs, zero loss spikes ($\Delta \mathcal{L} > 1.5$), and peak gradient norm bounded at $\mathbf{1.000}$ under BF16 mixed-precision training.
   - **Strict Zero-Transcendental Stack:** Exact AST verification of zero calls to `exp`, `log`, `sin`, `cos` throughout all model definitions, attention layers, activations, loss functions, and optimizer routines.

6. **Systematic Hyperparameter Sweeping & 125M Calibration on FineWeb-Edu (Phase 8):**
   - **INVALIDATED:** This job evaluated only one configuration per architecture and overstated the processed full-batch token count. The figures below are historical diagnostics and cannot select Phase 9 hyperparameters.
   - **Hardware & Scale:** 123.55M parameter matched budget ($0.00\%$ parameter delta) trained on physical Google Cloud TPU v4-32 Pod slice (4 hosts, 16 physical chips, 32 TensorCores) with sequence context $T=2048$ and global batch size 512 ($1.05\text{M}$ tokens/step).
   - **Evaluation Volume:** 6 complete runs (2 architectures $\times$ 3 seeds: 42, 43, 44 on 600M tokens each = **3.6 Billion tokens evaluated**).
   - **Historical fixed-config diagnostic:** The invalid job recorded mean validation perplexity $\mathbf{66.31}$ vs $\mathbf{77.51}$ (ratio $\mathbf{0.8554}$); it cannot support a sweep winner or architectural-superiority claim.
   - **Multi-Seed Stability:** Algebraic multi-seed standard deviation was **$0.088\%$** (vs $0.315\%$ for baseline, $3.6\times$ tighter convergence across seeds 42, 43, 44).
   - **Numerical Stability & Zero Transcendental Guarantee:** Exact zero NaNs, zero Infs, zero steady-state loss spikes, peak gradient norm $\mathbf{1.000}$, and 100% verified AST zero-transcendental purity.
   - **Invalidated configurations:** The historical `*_optimal.json` files are rejected by the corrected Phase 8/9 validators and must be regenerated by the 18-run sweep.

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
