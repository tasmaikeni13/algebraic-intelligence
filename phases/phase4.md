# Phase 4: Algebraic Loss Functionals & Information Metrics (OACE / $\mathcal{L}_{1/8}$)

Start only after Phase 3 PASS. Read `theory.md`, `formal/README.md`, `formal/AlgebraicTheory/Loss.lean`, Phase 3 evidence in `results/phase3/`, and `phases/README.md` completely before executing. Execute the shared failure-repair loop until all gates pass.

---

## 1. Objective, Scientific Hypothesis & Competing Models

Eliminate the natural logarithm $\ln(x)$ and Shannon entropy from neural training objectives:
$$\textbf{"Can an algebraic divergence train neural distributions without logarithmic gradient poles?"}$$

### Competing Hypotheses:
- **$H_1$ (Algebraic Hypothesis):** The Octo-Algebraic Cross-Entropy $\mathcal{L}_{1/8}(p_k) = 8(p_k^{-1/8} - 1)$ evaluated via 3 sequential hardware $\operatorname{rsqrt}$ operations is a strictly proper scoring rule that eliminates the infinite gradient singularity of cross-entropy near simplex boundaries ($p \to 0$), matches the Riemannian Fisher information metric of KL divergence, and stabilizes gradient descent under extreme label noise.
- **$H_0$ (Transcendental Baseline Hypothesis):** The logarithmic cross-entropy $\mathcal{L}_{\text{CE}} = -\ln p$ is uniquely derived from maximum likelihood; any algebraic replacement will distort prediction calibration, suffer from slow convergence, or fail on multi-class language modeling.

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 The $\alpha$-Algebraic Cross-Entropy Family
For probability distribution $\mathbf{p} \in \operatorname{int}\Delta^{K-1}$ and target index $k$:
$$\mathcal{L}_{1/8}(p_k) = 8\left( p_k^{-1/8} - 1 \right)$$
Evaluation proceeds strictly via 3 hardware square-root/rsqrt operations:
$$z_1 = \operatorname{rsqrt}(p_k) = p_k^{-1/2}, \quad z_2 = \operatorname{rsqrt}(z_1^{-1}) = p_k^{-1/4}, \quad z_3 = \operatorname{rsqrt}(z_2^{-1}) = p_k^{-1/8}$$
with zero log calls.

### 2.2 Gradient Boundedness & Fisher Curvature
- **Gradient:** $\frac{d}{d\hat{s}_j}\mathcal{L}_{1/8} = -8 w_j p_k^{-1/8}(\delta_{kj} - p_j)$, with magnitude at $p_k = 10^{-9}$ bounded by $8 (10^{-9})^{-1/8} \approx 104.98$ (algebraic growth vs. infinite logarithmic pole $1/p_k = 10^9$).
- **Pearson $\chi^2$ Equivalence:** For target distribution $\mathbf{y}$, the algebraic divergence $D_A(\mathbf{y} \| \mathbf{p}) = \sum \frac{y_i^2}{p_i} - 1$ is non-negative and convex, with Hessian $H(D_A)|_{\mathbf{p}=\mathbf{y}} = 2 H(D_{\text{KL}})|_{\mathbf{p}=\mathbf{y}}$.

---

## 3. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Loss.lean` with zero errors under `/root/.elan/bin/lake build`:

1. `pearson_chi_sq_expansion`:
   $$\forall y, p \in \mathbb{R}, \ p \neq 0 \implies \frac{(y - p)^2}{p} = \frac{y^2}{p} - 2y + p$$
2. `pearson_divergence_nonneg`:
   $$\forall y, p \in \mathbb{R}, \ p > 0 \implies \frac{(y - p)^2}{p} \geq 0$$
3. `pearson_zero_iff_equal`:
   $$\frac{(y - p)^2}{p} = 0 \iff y = p \quad \text{for } p > 0$$

---

## 4. Deep Empirical & Monte Carlo Simulation Gate

Execute the Phase 4 test suite in `analysis/verify_algebraic_primitives.py`:

| Evaluation Dimension | Experimental Protocol | Success Criterion / Bound |
| :--- | :--- | :--- |
| **Monte Carlo Label Noise Stress Test** | $10^5$ trials under symmetric label noise $\epsilon \in [0.0, 0.3]$ | Gradient variance $\operatorname{Var}(\nabla \mathcal{L}_{1/8}) \le 0.50 \times \operatorname{Var}(\nabla \mathcal{L}_{\text{CE}})$ |
| **Simplex Boundary Stability** | Evaluate gradient across $p_k \in [10^{-9}, 1 - 10^{-9}]$ | Zero NaNs, zero Infs, bounded gradient |
| **Fisher Information Ratio** | Hessian ratio $H(D_A) / H(D_{\text{KL}})$ at $\mathbf{p} = \mathbf{y}$ | Exactly $[2.0, 2.0, \dots, 2.0]$ |
| **Strict Propriety & Monotonicity** | Verify $\frac{\partial \mathcal{L}_{1/8}}{\partial p_k} < 0$ across $10^5$ samples | Monotonically decreasing on $(0, 1]$ |
| **Minimum Value** | Evaluate $\min_{p_k \in (0, 1]} \mathcal{L}_{1/8}(p_k)$ | Exactly $0.000000$ at $p_k = 1.0$ |
| **Zero Transcendental Audit** | Grep of loss module for `log`, `ln`, `cross_entropy` | Exactly $0$ occurrences |

---

## 5. Autonomous Failure Ledger & Self-Correction Playbook

- **Symptom: Gradient overflow when probability approaches zero ($p_k \to 0$):**
  - *Root Cause:* Division by zero in unconstrained probability logits.
  - *Correction:* Confirm that A-Softmax attention sink $\Omega > 0$ provides a strictly positive mathematical lower bound on denominator.
- **Symptom: Optimization step size too small relative to standard CE:**
  - *Root Cause:* Constant scale factor mismatch in $\mathcal{L}_{1/8}$.
  - *Correction:* Scale OACE loss by rational calibration factor $\gamma = 2.0$.

---

## 6. Passing Gate Checklist

- [ ] `formal/AlgebraicTheory/Loss.lean` compiles with 0 errors via `/root/.elan/bin/lake build`.
- [ ] $10^5$-trial Monte Carlo label noise simulation proves $\le 50\%$ gradient variance vs. Cross-Entropy.
- [ ] Simplex boundary evaluation confirms zero gradient singularities at $p_k = 10^{-9}$.
- [ ] Fisher information equivalence ratio is identically $2.0$.
- [ ] OACE loss strictly proper, monotonic, and zero at $p_k = 1.0$.
- [ ] Zero log calls verified in loss codebase.
- [ ] `results/phase4/PASS.md` satisfies the shared PASS record contract.
