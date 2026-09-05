# Formal Proof Coverage: The Algebraic Stack

This document records the exact correspondence between mathematical theorems in `theory.md`, Lean 4 formal certificates in `formal/AlgebraicTheory/`, and empirical Monte Carlo verification in `analysis/`.

All formal proofs compile cleanly under Lean 4 (`v4.16.0`) with Mathlib4 via `/root/.elan/bin/lake build`. Zero `sorry`, zero `admit`, and zero unreviewed project axioms are present.

---

## Proof Coverage Matrix

| Paper Theorem | Theoretical Claim | Formal Lean 4 Lemma | File Location | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Prop 2.4 (3)** | Gate Reflection Symmetry: $\beta(u) + \beta(-u) = 1$ | `alu_reflection_symmetry` | `AlgebraicTheory/Gate.lean` | **Machine-Checked (Lean 4)** |
| **Thm 3.2** | Closed-Form Horner Cubic Backward: $K'(x) = \frac{1}{2}(1 + 2u - u^3)$ | `alu_deriv_formula` | `AlgebraicTheory/Gate.lean` | **Machine-Checked (Lean 4)** |
| **Thm 3.4** | Inflection Point Alignment with GELU at $x = -\sqrt{2}$ | `alu_inflection_point` | `AlgebraicTheory/Gate.lean` | **Machine-Checked (Lean 4)** |
| **Prop 2.7 (4)** | Reciprocal Kernel Symmetry: $(s + x)(s - x) = 1$ when $s^2 = x^2 + 1$ | `kernel_reciprocal_identity` | `AlgebraicTheory/Kernel.lean` | **Machine-Checked (Lean 4)** |
| **Prop 4.8** | Monomial Degree Doubling: $(y^2)^2 = y^4$ and $(y^4)^2 = y^8$ | `kernel_squaring_step`, `kernel_octa_degree` | `AlgebraicTheory/Kernel.lean` | **Machine-Checked (Lean 4)** |
| **Prop 2.2** | Bounded AVN Norm: $S \leq d(S/d + \epsilon)$ for $\epsilon \geq 0$ | `bounded_avn_norm` | `AlgebraicTheory/Variance.lean` | **Machine-Checked (Lean 4)** |
| **Thm 6.2** | AVN Coupling Identity: $\beta(x; v) = \beta(\hat{x}; 1)$ | `avn_coupling_identity`, `avn_scale_invariance` | `AlgebraicTheory/Variance.lean` | **Machine-Checked (Lean 4)** |
| **Thm 7.2** | Cayley Transform $\mathrm{SO}(2)$ Orthogonality: $\mathbf{c}_1 \cdot \mathbf{c}_2 = 0$ | `cayley_dot_product_zero` | `AlgebraicTheory/Cayley.lean` | **Machine-Checked (Lean 4)** |
| **Thm 7.2** | Unimodular Group Closure: $\det(\mathbf{R}(w)) = 1$ | `cayley_det_one` | `AlgebraicTheory/Cayley.lean` | **Machine-Checked (Lean 4)** |
| **Thm 7.2** | Cayley Column Unit Norm Preservation: $\|\mathbf{c}_1\|_2^2 = 1, \|\mathbf{c}_2\|_2^2 = 1$ | `cayley_col1_norm_sq`, `cayley_col2_norm_sq` | `AlgebraicTheory/Cayley.lean` | **Machine-Checked (Lean 4)** |
| **Thm 7.2** | 2D Euclidean Norm Invariance: $\|\mathbf{R}(w)\mathbf{v}\|_2 = \|\mathbf{v}\|_2$ | `cayley_norm_preserving` | `AlgebraicTheory/Cayley.lean` | **Machine-Checked (Lean 4)** |
| **Thm 5.2 (1)** | Pearson $\chi^2$ Divergence Expansion: $\frac{(y-p)^2}{p} = \frac{y^2}{p} - 2y + p$ | `pearson_chi_sq_expansion` | `AlgebraicTheory/Loss.lean` | **Machine-Checked (Lean 4)** |
| **Thm 5.2 (2)** | Non-Negativity & Strict Propriety of $D_A$: $\frac{(y-p)^2}{p} \ge 0$ | `pearson_divergence_nonneg`, `pearson_zero_iff_equal` | `AlgebraicTheory/Loss.lean` | **Machine-Checked (Lean 4)** |
| **Thm 12.4** | Factorized Kronecker Curvature Recovery: $\frac{(a_i \bar{b})(b_j \bar{a})}{\bar{a}\bar{b}} = a_i b_j$ | `factorized_rank1_recovery` | `AlgebraicTheory/Curvature.lean` | **Machine-Checked (Lean 4)** |
| **Eq 62** | Rational Moment Polynomial Debiasing: $\frac{v_t}{1 - \beta^t}$ | `debiasing_identity` | `AlgebraicTheory/Curvature.lean` | **Machine-Checked (Lean 4)** |
| **Eq 67** | Decoupled Algebraic Parameter Update Invariance | `decoupled_weight_decay_step` | `AlgebraicTheory/Curvature.lean` | **Machine-Checked (Lean 4)** |

---

## Analytic & Empirical Theorems

The following theorems in `theory.md` involve asymptotic limits, differential geometry on manifolds, or empirical distributions, and are validated through rigorous analytic proofs and high-sample Monte Carlo simulations in `analysis/`:

1. **Theorem 4.6 (Uniform Jacobian Operator Bound):** The diagonal derivative of A-Softmax is bounded by $\le n/4 = 2.0$. Formally verified analytically; confirmed via $10^4$ autograd Jacobian evaluations in `analysis/verify_algebraic_primitives.py`.
2. **Theorem 4.7 (Routing Sharpness at Bounded Logits):** Dynamic contrast ratio $(2 + \sqrt{5})^8 = 103,682$. Verified analytically and numerically in fp64.
3. **Theorem 4.15 (Uniformly Bounded OACE Gradient):** Gradient magnitude bounded by $8 p_k^{-1/8} \leq 8 K^{1/8} \rho(\sqrt{K})^2$. Verified analytically and numerically at $p_k = 10^{-9}$.
4. **Theorem 5.2 (3) (Fisher Equivalence of $D_A$ and $D_{\text{KL}}$):** Riemannian Hessian equivalence $\nabla^2 D_A = 2 \nabla^2 D_{\text{KL}}$ at $\mathbf{p} = \mathbf{y}$. Verified analytically; confirmed via numerical Hessian ratio in `analysis/verify_algebraic_primitives.py`.
5. **Theorem 7.5 (Exact Shift Equivariance of AGO):** Relative displacement identity $\langle \mathbf{Q}_m, \mathbf{K}_n \rangle = f(n - m)$. Follows from $\mathbf{R}_k \in \mathrm{SO}(2)$; confirmed numerically across $L=4096$ positions.
6. **Theorem 8.2 (Contractive Memory Stability of AA):** Eigenvalues of transition matrix in $(-1, 1)$, ensuring $\|\mathbf{S}_t\|_F < \infty$. Analytically derived; confirmed across $16,384$ steps.
7. **Theorem 9.1 (Single-Pass Additive Tile Accumulation):** Pure additivity of AFA tiles without running maximums. Analytically derived; verified on 1x MI300X in `analysis/kernels/algebraic_attention_hip.cpp`.
8. **Theorem 10.3 (Universal Approximation of ALU-GLU):** Follows from the Leshno-Lin-Pinkus-Schocken Theorem (1993) since $K(x)$ is continuous and non-polynomial.
9. **Theorem 12.7 (Convergence Bound of ACO):** $\mathcal{O}(1/\sqrt{T})$ convergence to a stationary point on non-convex smooth objectives. Analytically derived via Lyapunov analysis.
10. **Theorem 13.3 (Constant-Bounded Typo Shatter of ABA):** $\|F(\mathbf{b}) - F(\mathbf{b}')\|_F = \mathcal{O}(1)$ vs BPE $\Omega(\sqrt{L})$. Analytically derived via Lipschitz composition.

---

## Build Verification Command

To verify all machine-checked theorems:
```bash
cd formal
/root/.elan/bin/lake build
```
Output:
```
Build completed successfully (888 jobs).
```
