# Phase 3 PASS — Algebraic Geometric Oscillators & Shift Equivariance (AGO)

Completed 2026-09-16 on the four-host, 16-chip Cloud TPU v4 Pod slice (`my-tpu-v4` in `us-central2-b`). All formal Lean 4 proof certificates, CPU empirical bounds, zero-transcendental AST purity audits, out-of-distribution associative recall generalizations, and distributed TPU hardware gates pass with zero failures.

---

## 1. Gate Inventory & Direct Evidence

| Gate | Scientific Requirement | Empirical Outcome | Direct Evidence |
| :--- | :--- | :--- | :--- |
| **Lean Proof Certificates** | Clean compilation of `formal/AlgebraicTheory/Cayley.lean`; 0 sorry, 0 admit, 0 axioms | 1526 jobs compiled cleanly; 0 warnings | [`formal/lean-build.log`](lean-build.log), [`formal/AlgebraicTheory/Cayley.lean`](../../formal/AlgebraicTheory/Cayley.lean), `metrics.json:formal` |
| **Zero-Transcendental Axiom** | 0 occurrences of `sin`, `cos`, `tan`, `pow`, `exp`, `log`, or complex numbers in production code | AST walk & token audit clean (0 violations, 0 regex hits) | `metrics.json:purity`, [`src/attention.py`](../../src/attention.py#L90-L240) |
| **Algebraic Unimodularity & Orthogonality** | $\|\det(\mathbf{R}(w)) - 1.0\| \le 1.0 \times 10^{-15}$ and $\|\mathbf{c}_1 \cdot \mathbf{c}_2\| \le 1.0 \times 10^{-15}$ across $10^5$ samples | $\max \|\det - 1\| = 4.44 \times 10^{-16}$; $\max \|\mathbf{c}_1 \cdot \mathbf{c}_2\| = 0.0$ | `metrics.json:determinant_and_orthogonality`, `tpu/metrics.json:unimodularity_and_orthogonality` |
| **Long-Context Shift Equivariance** | Shift equivariance error $\le 1.0 \times 10^{-6}$ across context $L = 4096$ | $\max \text{error} = 2.10 \times 10^{-14} \le 1.0 \times 10^{-6}$ (2.1M pairs evaluated) | `metrics.json:shift_equivariance`, `tpu/metrics.json:shift_equivariance` |
| **Relative Attention Dot Product** | Relative dot product error $\le 1.0 \times 10^{-6}$ over $10^5$ random query-key pairs | $\max \text{error} = 2.34 \times 10^{-13} \le 1.0 \times 10^{-6}$ | `metrics.json:relative_dot_product` |
| **Cumulative Norm Conservation** | Sequential rotation norm drift $\le 1.0 \times 10^{-6}$ across $m \in [1, 8192]$ | $\max \text{drift} = 2.22 \times 10^{-16} \le 1.0 \times 10^{-6}$ | `metrics.json:norm_conservation`, `tpu/metrics.json:norm_conservation` |
| **OOD Associative Recall Generalization** | Retrieval accuracy $\ge 95.0\%$ at $L=1024$ and $L=2048$ when trained on $L=256$ | $100.0\%$ at $L=1024$; $99.5\%$ at $L=2048$ | `metrics.json:associative_recall` |
| **TPU Numerical Parity** | Parity against float64 CPU oracle across 20 configurations (FP32/BF16, 5 lengths, 2 head dims) | 20 / 20 passed on 16 TPU v4 cores | `tpu/metrics.json:parity` |
| **TPU Synchronized Throughput** | Sustained throughput $\ge 90.0\%$ vs standard trigonometric RoPE on TPU across context lengths | 8 / 8 configurations passed ($98.6\% - 107.7\%$) | `tpu/metrics.json:benchmarks`, [`tpu/latencies.json`](tpu/latencies.json) |

---

## 2. Distributed TPU Throughput & Latency

Evaluated across all 16 physical TPU v4 chips on `my-tpu-v4` (topology $2 \times 2 \times 4$, 4 worker hosts) with 10 warmups and 100 synchronized repetitions per configuration:

| Dtype | Sequence Length $L$ | Global Shape | Forward Throughput Ratio (RoPE / AGO) | Forward + Backward Throughput Ratio | Gate Status ($\ge 90\%$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **float32** | 512 | `(16, 512, 8, 64)` | 1.007 | 1.029 | **PASS** |
| **float32** | 1024 | `(16, 1024, 8, 64)` | 0.997 | 1.021 | **PASS** |
| **float32** | 2048 | `(16, 2048, 8, 64)` | 0.993 | 1.046 | **PASS** |
| **float32** | 4096 | `(16, 4096, 8, 64)` | 1.000 | 1.064 | **PASS** |
| **bfloat16** | 512 | `(16, 512, 8, 64)` | 0.993 | 1.020 | **PASS** |
| **bfloat16** | 1024 | `(16, 1024, 8, 64)` | 0.986 | 1.030 | **PASS** |
| **bfloat16** | 2048 | `(16, 2048, 8, 64)` | 1.001 | 1.056 | **PASS** |
| **bfloat16** | 4096 | `(16, 4096, 8, 64)` | 1.003 | 1.077 | **PASS** |

On TPU v4, Algebraic Geometric Oscillators achieve matching or slightly superior throughput to trigonometric RoPE because the Cayley rotation block compiles into 4 parallel Fused Multiply-Add (`FMA`) instructions per 2D coordinate pair on the TPU VMU vector execution units, with zero transcendental function approximation overhead.

---

## 3. Large-Scale Empirical Studies

### Determinant & Orthogonality Study ($10^5$ Random Samples)
$$\mathbf{R}(w) = \frac{1}{1 + w^2} \begin{pmatrix} 1 - w^2 & -2w \\ 2w & 1 - w^2 \end{pmatrix}$$
- Sample domain: $w \in [10^{-5}, 100.0]$
- Max determinant error $\|\det(\mathbf{R}) - 1.0\|$: $4.4409 \times 10^{-16} \le 1.0 \times 10^{-15}$
- Max orthogonality error $\|\mathbf{c}_1 \cdot \mathbf{c}_2\|$: $0.0 \le 1.0 \times 10^{-15}$
- Column 1 norm error: $4.4409 \times 10^{-16} \le 1.0 \times 10^{-15}$
- Column 2 norm error: $4.4409 \times 10^{-16} \le 1.0 \times 10^{-15}$

### Long-Context Shift Equivariance ($L = 4096$)
- Pairs evaluated: $2,113,568$
- Max shift equivariance error: $2.0983 \times 10^{-14} \le 1.0 \times 10^{-6}$
- Max cosine block error: $1.5432 \times 10^{-14} \le 1.0 \times 10^{-6}$
- Max sine block error: $2.0983 \times 10^{-14} \le 1.0 \times 10^{-6}$

### Relative Attention Dot Product ($10^5$ Query-Key Pairs)
$$\langle \mathbf{R}(m)\mathbf{q}, \mathbf{R}(n)\mathbf{k} \rangle = \mathbf{q}^\top \mathbf{R}(m)^\top \mathbf{R}(n) \mathbf{k} = \mathbf{q}^\top \mathbf{R}(n - m) \mathbf{k}$$
- Context length: $L = 4096$, Dimension: $64$
- Max relative error: $2.3448 \times 10^{-13} \le 1.0 \times 10^{-6}$
- 95% Confidence Interval of error: $[1.5747 \times 10^{-14}, 1.5951 \times 10^{-14}]$

### Cumulative Norm Conservation ($L = 8192$)
- Max sequential composition drift: $2.2204 \times 10^{-16} \le 1.0 \times 10^{-6}$
- 95% Confidence Interval of drift: $[1.3496 \times 10^{-17}, 1.5886 \times 10^{-17}]$
- Re-normalization per step via algebraic radical `rsqrt(c^2 + s^2)` suppresses cumulative error to floating-point machine precision.

### Out-of-Distribution Associative Recall Study
- Sequence length trained: $L = 256$
- In-distribution accuracy ($L = 256$): $100.0\%$
- $4\times$ Context generalization ($L = 1024$): $100.0\% \ge 95.0\%$
- $8\times$ Context generalization ($L = 2048$): $99.5\% \ge 95.0\%$

---

## 4. Formal Lean 4 Verification Summary

File: [`formal/AlgebraicTheory/Cayley.lean`](../../formal/AlgebraicTheory/Cayley.lean)  
Toolchain: **Lean 4.34.0-rc2** (pinned by `formal/lake-manifest.json` and `formal/lean-toolchain`).

The following formal certificates have been verified with 0 warnings, 0 `sorry`, 0 `admit`, and 0 axioms:
1. `cayley_pythagorean_identity`: $(1 - w^2)^2 + (2w)^2 = (1 + w^2)^2$ for all $w \in \mathbb{R}$.
2. `cayley_column_norm_one`, `cayley_col1_norm_sq`, `cayley_col2_norm_sq`: Exact unit norm conservation for rotation column vectors when $1 + w^2 \neq 0$.
3. `cayley_columns_orthogonal`, `cayley_dot_product_zero`: Orthogonality $\mathbf{c}_1 \cdot \mathbf{c}_2 = 0$.
4. `cayley_determinant_one`, `cayley_det_one`: Unimodularity $\det(\mathbf{R}(w)) = 1$, certifying $\mathrm{SO}(2)$ Lie group closure.
5. `cayley_norm_preserving`, `cayley_rational_norm_preserving`: 2D Euclidean norm invariance $\|\mathbf{R}(w)\mathbf{v}\|_2 = \|\mathbf{v}\|_2$.
6. `cayley_composition_norm`: Preservation of unit norm under rotational composition.
7. `cayley_shift_equivariance`: Exact relative shift-equivariance algebraic identity $\langle \mathbf{R}_1 \mathbf{q}, \mathbf{R}_2 \mathbf{k} \rangle = \mathbf{q}^\top (\mathbf{R}_1^\top \mathbf{R}_2) \mathbf{k}$.

---

## 5. Architectural Implementation Details

Production implementation in [`src/attention.py`](../../src/attention.py):
- `CayleyRotary`: Structured NamedTuple container providing `(c, s)` parameter tables and `.matrix` representation.
- `build_cayley_rotary_matrix`: JIT-compiled recurrence precomputing Cayley rotation parameters without transcendentals. Employs high-precision recurrence (`calc_dtype = jnp.float32`) with algebraic radical re-normalization `r = jax.lax.rsqrt(c^2 + s^2)` before optional casting to `bfloat16`.
- `apply_ago_rotations`: Native vectorized tensor rotation supporting 2D, 3D, and 4D layouts via 4 FMAs per channel pair. Zero trigonometric or non-integer power calls.
