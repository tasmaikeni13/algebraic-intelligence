# Phase 1 — Mathematical Oracle, Calibration, & Pathology Gate

Start only after Phase 0 PASS. Read `theory.md`, Phase 0 artifacts in `results/phase0/`, and `phases/AUTONOMY_PROTOCOL.md` completely before executing. Execute the mandatory failure-repair loop until all current and inherited gates pass.

The scientific objective of this phase is to **falsify the exact mathematical equations of the Algebraic Stack** under extreme numerical stress, boundary conditions, and ill-conditioning before learned representations or optimization scale can mask analytical errors.

---

## 1. Controlled Mathematical Sweeps & Stress Experiments

Construct a comprehensive pinned verification suite (`analysis/verify_algebraic_primitives.py`) that systematically sweeps:
- **Dimensionality:** Feature dimensions $d \in \{16, 32, 64, 128, 256, 512, 1024\}$, attention head dimensions $d_k \in \{16, 32, 64, 128\}$, and context lengths $L \in \{64, 256, 1024, 2048, 4096, 8192\}$;
- **Conditioning:** Ill-conditioned covariance matrices and linear systems with condition numbers $\kappa \in [1, 10^6]$;
- **Logit Distributions:** Well-conditioned Gaussian logits, heavy-tailed Student-t/Cauchy logits, near-duplicate logits, and sub-byte quantized logits;
- **Simplex Boundary Extremes:** Target probabilities spanning nine orders of magnitude: $p_k \in [10^{-9}, 1 - 10^{-9}]$;
- **Input Variance Spreads:** Input feature norms $\|\mathbf{x}\|_2 / \sqrt{d} \in [10^{-4}, 10^4]$;
- **Precision Levels:** Compare fp64 (CPU oracle), fp32, and bf16 on identical quantized inputs.

---

## 2. Definitive Primitive Verifications

### 2.1 The Algebraic Linear Unit (ALU) & Gate
1. **Horner Cubic Backward Exactness:** Compare the forward cache Horner derivative $K'(x) = 0.5 + u(1.0 - 0.5 u^2)$ against double-precision autograd across $10^6$ uniform samples $x \in [-100, 100]$. The absolute discrepancy must not exceed condition-aware tolerance ($< 5.0 \times 10^{-16}$).
2. **Lipschitz Constant Supremum:** Certify empirically that $\sup_{x \in \mathbb{R}} |K'(x)|$ converges to the theoretical bound $L_K = \frac{1}{2}(1 + \frac{4\sqrt{6}}{9}) \approx 1.044331$, saturated at $u^* = \sqrt{2/3}$ ($x^* = \sqrt{2}$).
3. **Inflection Point Alignment:** Verify that $K''(x) = 0$ if and only if $x = -\sqrt{2}$, matching the inflection point of GELU. Verify that $K''(-\sqrt{2}) = 0.0$ to within floating-point epsilon.
4. **Gate Reflection Symmetry:** Confirm $\beta(u) + \beta(-u) = 1.0$ holds identically across all evaluated points (residual $= 0.0$).

### 2.2 Algebraic Variance Normalization (AVN)
1. **Variance Normalization:** Verify that $\operatorname{Var}(\operatorname{AVN}(\mathbf{x})) \in [0.9999, 1.0001]$ across $10^6$ samples for input standard deviations $\sigma \in [0.1, 10.0]$.
2. **Coupling Identity:** Verify that $\beta(x; v) = \beta(\hat{x}; 1)$ holds identically with maximum absolute error $< 1.0 \times 10^{-15}$, confirming downstream gates can evaluate dynamically using the upstream AVN variance scalar $\tau$.
3. **Closed-Form Backward Pass:** Verify that the analytic projection $\tau (\mathbf{g} - \frac{\langle \mathbf{g}, \hat{\mathbf{x}} \rangle}{d}\hat{\mathbf{x}})$ matches PyTorch autograd backward with relative error $< 1.0 \times 10^{-7}$.

### 2.3 Algebraic Softmax (A-Softmax)
1. **Uniform 2-Lipschitz Operator Bound:** Compute the full $K \times K$ Jacobian matrix $\mathbf{J}_{ij} = \partial p_i / \partial \hat{s}_j$ across $10^4$ trials with random inputs. Confirm that the diagonal derivative satisfies $\max_j |\mathbf{J}_{jj}| \leq 2.000$ (theorized bound $n/4$ for $n=8$) and that the maximum spectral norm $\|\mathbf{J}\|_2 \leq 2.0$.
2. **Sharp Routing Contrast on Bounded Logits:** Confirm that for $\hat{s}_1 = 2.0$ and $\hat{s}_2 = 0.0$, the routing ratio matches $(2 + \sqrt{5})^8 = 103,682$ to 6 significant figures.
3. **Rational Attention Sink ($\Omega$):** Verify that setting $\Omega > 0$ strictly bounds $\sum_{i=1}^K p_i \leq 1.0$ and prevents division by zero when all logits are negative.
4. **FP4 Quantization Stability:** Measure output distribution displacement under logit quantization noise $\sigma = 0.05$. Confirm that A-Softmax exhibits $\ge 100\times$ lower displacement than exponential Softmax.

### 2.4 OACE ($\mathcal{L}_{1/8}$) & Algebraic Divergence ($D_A$)
1. **Simplex Boundary Stability:** Evaluate OACE loss and its analytic gradient at $p_k = 10^{-9}$. Confirm zero NaNs, zero Infs, and that the gradient magnitude matches $8 p_k^{-1/8} = 8 (10^{-9})^{-1/8} \approx 104.98$ (algebraic stability vs. logarithmic gradient overflow $1/p_k = 10^9$).
2. **Strict Propriety & Monotonicity:** Confirm that $\mathcal{L}_{1/8}(p_k) \geq 0$ for all $p_k \in (0, 1]$, with $\mathcal{L}_{1/8} = 0$ iff $p_k = 1.0$, and that $\frac{d}{dp_k}\mathcal{L}_{1/8} < 0$.
3. **Pearson $\chi^2$ Expansion:** Verify that $D_A(\mathbf{y} \| \mathbf{p}) = \sum y_i^2 / p_i - 1 = \sum (y_i - p_i)^2 / p_i$ across $10^5$ distributions.
4. **Fisher Metric Equivalence:** Compute the numerical Hessian of $D_A$ and $D_{\text{KL}}$ at $\mathbf{p} = \mathbf{y}$. Confirm that $\nabla^2 D_A = 2 \nabla^2 D_{\text{KL}}$ with diagonal ratio $[2.0, 2.0, \dots, 2.0]$.

### 2.5 Algebraic Geometric Ordering (AGO)
1. **Unimodularity & Orthogonality:** For frequency parameters $w \in [10^{-5}, 10^2]$, verify $|\det(\mathbf{R}(w)) - 1.0| \le 1.0 \times 10^{-15}$ and column inner product $|\mathbf{c}_1 \cdot \mathbf{c}_2| \le 1.0 \times 10^{-15}$.
2. **Norm Conservation Over Long Horizons:** Propagate 2D vectors under repeated Cayley multiplication $\mathbf{R}^m \mathbf{v}$ up to sequence length $m = 8192$. Confirm that $|\|\mathbf{R}^m \mathbf{v}\|_2 - \|\mathbf{v}\|_2| \leq 1.0 \times 10^{-6}$.
3. **Exact Relative Shift Equivariance:** For positions $m, n \in [0, 4096]$, compute $\langle \mathbf{Q}_m, \mathbf{K}_n \rangle$ and verify $\|\mathbf{R}_m^\top \mathbf{R}_n - \mathbf{R}_{n-m}\|_\infty \leq 1.0 \times 10^{-6}$.

### 2.6 Algebraic Curvature Optimizer (ACO)
1. **Kronecker Rank-1 Curvature Recovery:** Verify that for synthetic Kronecker outer products $\mathbf{G}^{\odot 2} = \mathbf{a} \mathbf{b}^\top$, the factorized estimator $\frac{(a_i \bar{b})(b_j \bar{a})}{\bar{a}\bar{b}}$ recovers $a_i b_j$ with zero error.
2. **Convergence on Ill-Conditioned Surfaces:** Test ACO on quadratic loss surfaces with condition number $\kappa = 1000$. Confirm $> 99.99\%$ loss reduction in 300 steps with zero divergence.
3. **ARDS Monotonicity:** Confirm that the rational decay schedule $\eta_t = \eta_0 \operatorname{rsqrt}(1 + \alpha ((t - T_{\text{warm}})/T_{\text{decay}})^2)$ is strictly monotonically decreasing for $t > T_{\text{warm}}$ and asymptotic to $\mathcal{O}(1/t)$.

---

## 3. Formal Lean 4 Requirements

Audit every theorem used in the numerical sweeps against `formal/AlgebraicTheory/`:
- `formal/AlgebraicTheory/Gate.lean`: `alu_reflection_symmetry`, `alu_deriv_formula`, `alu_inflection_point`;
- `formal/AlgebraicTheory/Kernel.lean`: `kernel_reciprocal_identity`, `kernel_squaring_step`, `kernel_octa_degree`;
- `formal/AlgebraicTheory/Variance.lean`: `bounded_avn_norm`, `avn_coupling_identity`, `avn_scale_invariance`;
- `formal/AlgebraicTheory/Cayley.lean`: `cayley_pythagorean_identity`, `cayley_col1_norm_sq`, `cayley_dot_product_zero`, `cayley_det_one`, `cayley_norm_preserving`;
- `formal/AlgebraicTheory/Loss.lean`: `pearson_chi_sq_expansion`, `pearson_divergence_nonneg`, `pearson_zero_iff_equal`;
- `formal/AlgebraicTheory/Curvature.lean`: `factorized_rank1_recovery`, `debiasing_identity`, `decoupled_weight_decay_step`.

Compile `formal/` via `/root/.elan/bin/lake build`. No `sorry`, `admit`, or unreviewed axioms are permitted.

---

## PASS Gates

- [ ] All exact fp64 identities match their condition-aware tolerances across all evaluated rows.
- [ ] Horner polynomial backward pass matches fp64 autograd with error $< 5.0 \times 10^{-16}$.
- [ ] Empirical ALU inflection point matches $-\sqrt{2}$ with residual $< 1.0 \times 10^{-15}$.
- [ ] A-Softmax diagonal Jacobian is strictly $\le 2.0$ across $10^4$ trials.
- [ ] A-Softmax exhibits $\ge 100\times$ lower output sensitivity to logit quantization noise than exponential Softmax.
- [ ] OACE loss gradient is strictly bounded at $p_k = 10^{-9}$ with zero NaNs or Infs.
- [ ] Algebraic Divergence Hessian satisfies $\nabla^2 D_A = 2 \nabla^2 D_{\text{KL}}$ with diagonal ratio $[2.0, 2.0, \dots, 2.0]$.
- [ ] AGO Cayley rotation preserves unit norm up to sequence length $m = 8192$ with drift $\le 1.0 \times 10^{-6}$.
- [ ] ACO converges on ill-conditioned quadratics ($\kappa = 1000$) with $> 99.99\%$ loss reduction in 300 steps.
- [ ] All Lean 4 proofs compile cleanly via `/root/.elan/bin/lake build`.
- [ ] All Phase 0 inherited gates pass.
- [ ] `results/phase1/PASS.md` is generated and committed, satisfying the shared PASS record contract.
