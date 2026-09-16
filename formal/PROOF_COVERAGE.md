# Formal proof coverage

The authoritative compiler and dependency versions are `lean-toolchain` and
`lake-manifest.json`. The root build covers all imported modules; the current
Phase 1 command, exit status, and no-placeholder audit are recorded in
`results/phase1/metrics.json`, with full output in `results/phase1/lean-build.log`.

## Phase 1 claims and exact certificates

| Claim | Lean theorem(s) | Scope |
| --- | --- | --- |
| Reflection symmetry | `alu_reflection_symmetry`, `gate_reflection_identity` | Exact polynomial identity for every real u |
| Cached derivative | `alu_deriv_formula`, `alu_polynomial_backward_identity` | Algebraic rearrangement of the derivative; does not formalize differentiation of the implemented floating-point function |
| Cache relation | `alu_cache_invertibility` | Assumes s²=x²+1 and us=x |
| Both inflections | `alu_inflection_iff` | Under the cache relation, 2−3u²=0 iff x²=2; negative-only claims were incorrect |
| Negative-tail repair | `alu_negative_tail_identity` | 1+u=r²/(1−u) when u²+r²=1 and denominator nonzero |
| Derivative bound | `alu_derivative_bound` | −0.05 ≤ Horner(u) ≤ 1.05 on [−1,1]; tight 1.044331… value remains analytic/numerical |
| Bounded normalization | `bounded_avn_norm`, `avn_bounded_norm` | S ≤ d(S/d+epsilon), with the stated sign assumptions |
| Coupling of coordinates | `avn_coupling_identity` | Squared-coordinate polynomial relation under tau²v=1 |
| Coupling of gates | `avn_gate_coupling` | Actual real-square-root coordinate identity, with tau>0 |
| Positive scale invariance | `avn_positive_scale_invariance` | alpha x / sqrt(alpha²v) = x / sqrt(v) for alpha>0 |
| Reciprocal scale cancellation | `avn_scale_invariance` | (tau/alpha)(alpha x)=tau x for alpha≠0 |
| Regularized second moment | `avn_regularized_moment` | m/(m+epsilon)=1−epsilon/(m+epsilon) |
| Centered variance | `avn_centered_variance` | tau²m−(tau mu)²=tau²(m−mu²) |
| Radial damping | `avn_radial_damping` | 1−m/(m+epsilon)=epsilon/(m+epsilon) |

The JAX VJPs, machine rounding, reductions, confidence intervals, and throughput
are checked by execution, not by these real-arithmetic proofs. In particular,
there is no formal theorem here proving arbitrary deep stacks preserve variance.

## Pre-existing later-phase certificates

These compile as inherited modules without new later-phase implementation work:

- `Kernel.lean`: reciprocal identity, octic power composition, and degree counting.
- `Cayley.lean`: rational column norms, orthogonality, determinant one, norm
  preservation, and composition under explicitly assumed unit norms.
- `Loss.lean`: scalar Pearson expansion, nonnegativity, equality condition, and
  the eighth-power chain. These do not prove OACE propriety or Fisher equivalence.
- `Curvature.lean`: rank-one factorization, debiasing cancellation, and decoupled
  decay algebra. These do not prove optimizer convergence.

The previous coverage document attributed broader unimplemented experiments and
nonexistent lemma names to these files. Later-phase empirical and analytic claims
in `theory.md` remain outside the Phase 1 completion claim.

## Phase 2 additions

`Kernel.lean` now also proves `attention_diagonal_factor`,
`attention_offdiagonal_factor`, and `attention_entry_bound` under explicit
probability and radical-factor assumptions, `kernel_sharpness_exact`, and
`attention_sink_mass`. These prove real-algebra bounds and identities; they do
not formalize differentiation, a 2-Lipschitz matrix norm, floating-point simplex
rounding, quantization superiority, or hardware performance. The numerical
Jacobian and VJP studies test the calculus implementation separately.

## Phase 3 additions

`Cayley.lean` formally certifies the foundational algebraic theorems for
Algebraic Geometric Oscillators (AGO) in $\mathrm{SO}(2)$:
- `cayley_pythagorean_identity`: $(1 - w^2)^2 + (2w)^2 = (1 + w^2)^2$ for all $w \in \mathbb{R}$.
- `cayley_column_norm_one`, `cayley_col1_norm_sq`, `cayley_col2_norm_sq`: exact unit norm conservation for rotation column vectors when $1 + w^2 \neq 0$.
- `cayley_columns_orthogonal`, `cayley_dot_product_zero`: column orthogonality $\mathbf{c}_1 \cdot \mathbf{c}_2 = 0$.
- `cayley_determinant_one`, `cayley_det_one`: unimodularity $\det(\mathbf{R}(w)) = 1$, proving $\mathrm{SO}(2)$ Lie group closure.
- `cayley_norm_preserving`, `cayley_rational_norm_preserving`: 2D Euclidean norm invariance $\|\mathbf{R}(w)\mathbf{v}\|_2 = \|\mathbf{v}\|_2$.
- `cayley_composition_norm`: preservation of unit norm under rotational composition.
- `cayley_shift_equivariance`: exact relative shift-equivariance algebraic identity $\langle \mathbf{R}_1 \mathbf{q}, \mathbf{R}_2 \mathbf{k} \rangle = \mathbf{q}^\top (\mathbf{R}_1^\top \mathbf{R}_2) \mathbf{k}$.

## Phase 4 additions

`Loss.lean` formally certifies the foundational algebraic theorems for Octic Algebraic Cross-Entropy (OACE) and Pearson $\chi^2$ Divergence:
- `pearson_chi_sq_expansion`: algebraic identity $(p - q)^2 / q = p^2/q - 2p + q$ for all $q > 0$.
- `pearson_divergence_expansion`: simplex sum identity $\sum_k \frac{(p_k - q_k)^2}{q_k} = \sum_k \frac{p_k^2}{q_k} - 1$ for distributions summing to 1.
- `pearson_divergence_nonneg`: non-negativity $D_P(p \| q) \ge 0$ as a sum of non-negative rational terms.
- `pearson_zero_iff_equal`: information metric fidelity $D_P(p \| q) = 0 \iff p = q$.
- `oace_power_chain`: hardware radical decomposition $p_k^{-1/8} = \left(\left(p_k^{-1/2}\right)^{-1/2}\right)^{-1/2}$, proving 3 sequential $\operatorname{rsqrt}$ operations evaluate the exact algebraic octic reciprocal.

## Phase 5 additions

`Curvature.lean` formally certifies the foundational algebraic theorems for Algebraic AdamW optimization and rational scheduling:
- `adamw_debiasing_identity`: exact polynomial moment debiasing $\frac{m}{1 - \beta^t} \cdot (1 - \beta^t) = m$, certifying that division by integer powers $1 - \beta^t$ recovers the unbiased moment.
- `adamw_decoupled_weight_decay`: decoupled algebraic parameter update invariance $w - \eta u - \eta \lambda w = (1 - \eta \lambda) w - \eta u$, proving parameter trajectories belong strictly to $\mathbb{Q}(\mathbf{W}_0, \mathbf{G}_1, \dots, \mathbf{G}_t, \sqrt{\cdot})$.
- `adamw_factorized_curvature_recovery`: exact algebraic recovery of separable curvature factors without transcendental logarithms or matrix exponentials.

## Phase 6 additions

`Kernel.lean` formally certifies the foundational algebraic theorems for Hardware-Fused Algebraic FlashAttention (AFA):
- `afa_additive_associativity`: single-pass additive tile accumulation associativity $(P_1 V_1 + P_2 V_2) = (P_1 V_1) + (P_2 V_2)$, proving partial attention blocks can be accumulated additively across tiles and mesh chips without online rescaling or inter-tile normalization barriers.
- `afa_scaling_invariance`, `afa_scaling_invariance_pos`: exact numerator-denominator scale invariance $(\alpha O) / (\alpha D) = O / D$ for $\alpha > 0$, certifying that distributed additive sums yield mathematically identical attention representations regardless of global or local normalization factorings.




