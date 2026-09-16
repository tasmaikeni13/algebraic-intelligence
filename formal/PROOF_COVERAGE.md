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
