# Lean 4 Formal Verification of the Algebraic Stack

This directory contains machine-checked Lean 4 proofs for the core mathematical theorems underpinning the **Algebraic Stack** (`AlgebraicTheory`).

All theorems are formally verified using Lean 4 (`v4.16.0`) and Mathlib4.

## Formally Verified Modules

The formal theory is structured modularly in `AlgebraicTheory/`:

1. **`Gate.lean` (`AlgebraicTheory.Gate`)**:
   - `alu_reflection_symmetry`: $\beta(u) + \beta(-u) = 1$ (exact partition of unity and reflection symmetry).
   - `alu_inv_symmetric`: Algebraic invertibility and inverse-squaring properties.
   - `alu_deriv_formula`: Polynomial derivative formula $\beta'(u) = \frac{1}{2}(1 + 2u - u^3)$.
   - `alu_inflection_point`: Non-vanishing gradient and inflection point dynamics at $u = -\sqrt{2}$.

2. **`Kernel.lean` (`AlgebraicTheory.Kernel`)**:
   - `kernel_reciprocal_identity`: $(s + x)(s - x) = s^2 - x^2 = 1$ when $s = \sqrt{1 + x^2}$ (reciprocal symmetry).
   - `kernel_squaring_step`: Monomial composition and degree doubling across successive squaring stages.
   - `kernel_octa_degree`: Exact degree progression $2 \to 4 \to 8$ for the octic algebraic kernel $\kappa_8(x)$.

3. **`Cayley.lean` (`AlgebraicTheory.Cayley`)**:
   - `cayley_pythagorean_identity`: $(1 - w^2)^2 + (2w)^2 = (1 + w^2)^2$.
   - `cayley_col1_norm_sq`, `cayley_col2_norm_sq`: Exact unit norm preservation for rotation columns.
   - `cayley_dot_product_zero`: Orthogonality of rotation column vectors $\mathbf{c}_1 \cdot \mathbf{c}_2 = 0$.
   - `cayley_det_one`: Strict unimodularity $\det(\mathbf{R}(w)) = 1$ (exact $\mathrm{SO}(2)$ Lie group membership).
   - `cayley_norm_preserving`: Invariance of Euclidean norm under algebraic Cayley rotation $\|\mathbf{R}(w)\mathbf{v}\| = \|\mathbf{v}\|$.

4. **`Loss.lean` (`AlgebraicTheory.Loss`)**:
   - `pearson_chi_sq_expansion`: Algebraic expansion $(y - p)^2 / p = y^2/p - 2y + p$.
   - `pearson_divergence_nonneg`: Non-negativity of Pearson $\chi^2$ divergence $(y - p)^2 / p \geq 0$ for $p > 0$.

5. **`Curvature.lean` (`AlgebraicTheory.Curvature`)**:
   - `factorized_rank1_recovery`: Exact recovery of rank-1 curvature tensor from row and column marginals $\frac{(a_i \bar{b})(b_j \bar{a})}{\bar{a}\bar{b}} = a_i b_j$.
   - `debiasing_identity`: Polynomial normalization identity $\frac{v_t}{1 - \beta^t}$.
   - `decoupled_weight_decay_step`: Decoupled algebraic weight decay formulation.

6. **`Variance.lean` (`AlgebraicTheory.Variance`)**:
   - `bounded_avn_norm`: $S \leq d(S/d + \epsilon)$ for $\epsilon \geq 0$ (AVN activation normalization bound).
   - `avn_coupling_identity`: Variance preservation under algebraic reciprocal scaling.

## Building and Verifying

To verify the proofs using `lake`:

```bash
cd formal
lake build
```

Expected output:
```
Build completed successfully (zero errors, zero warnings).
```
