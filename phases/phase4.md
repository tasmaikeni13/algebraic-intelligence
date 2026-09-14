# Phase 4: Algebraic Loss Functionals & Information Metrics (OACE / $\mathcal{L}_{1/8}$)

Start only after Phase 3 PASS. Read `theory.md`, `formal/README.md`, `formal/AlgebraicTheory/Loss.lean`, Phase 3 evidence in `results/phase3/`, and `phases/README.md` completely before executing. Execute the shared adaptive failure-repair loop until all gates pass.

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
Evaluation on the TPU v4 VMU proceeds strictly via 3 hardware square-root/rsqrt operations:
$$z_1 = \operatorname{rsqrt}(p_k) = p_k^{-1/2}, \quad z_2 = \operatorname{rsqrt}(z_1^{-1}) = p_k^{-1/4}, \quad z_3 = \operatorname{rsqrt}(z_2^{-1}) = p_k^{-1/8}$$
with zero log calls.

### 2.2 Gradient Boundedness & Fisher Curvature
- **Gradient:** $\frac{d}{d\hat{s}_j}\mathcal{L}_{1/8} = -8 w_j p_k^{-1/8}(\delta_{kj} - p_j)$, with magnitude at $p_k = 10^{-9}$ bounded by $8 (10^{-9})^{-1/8} \approx 104.98$ (algebraic growth vs. infinite logarithmic pole $1/p_k = 10^9$).
- **Pearson $\chi^2$ Equivalence:** For target distribution $\mathbf{y}$, the algebraic divergence $D_A(\mathbf{y} \| \mathbf{p}) = \sum \frac{y_i^2}{p_i} - 1$ is non-negative and convex, with Hessian $H(D_A)|_{\mathbf{p}=\mathbf{y}} = 2 H(D_{\text{KL}})|_{\mathbf{p}=\mathbf{y}}$.

---

## 3. Implementation Target: JAX / TPU v4 Architecture

Instruct the creation and verification of the following files targeting the 16 TPU v4 Pod:
1. **`src/loss.py`**:
   - `oace_loss(probabilities, targets, gamma=2.0)`: JAX vectorized implementation of $\mathcal{L}_{1/8}$ using 3 sequential hardware $\operatorname{rsqrt}$ operations.
   - `pearson_divergence(p, q)`: Exact Pearson $\chi^2$ algebraic divergence in JAX.
   - Closed-form analytical VJP preventing any transcendental decomposition.
2. **`tests/test_loss.py`**:
   - Gradient boundedness tests at simplex boundary $p_k = 10^{-9}$.
   - Equivalence test verifying Riemannian Fisher metric $H(D_A) = 2 H(D_{\text{KL}})$.
   - AST audit confirming zero calls to `jax.nn.log_softmax`, `optax.softmax_cross_entropy`, or `log`.

---

## 4. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Loss.lean` with zero errors under `/root/.elan/bin/lake build`:

1. `pearson_chi_sq_expansion`:
   $$\forall y, p \in \mathbb{R}, \ p \neq 0 \implies \frac{(y - p)^2}{p} = \frac{y^2}{p} - 2y + p$$
2. `pearson_divergence_nonneg`:
   $$\forall y, p \in \mathbb{R}, \ p > 0 \implies \frac{(y - p)^2}{p} \geq 0$$
3. `pearson_zero_iff_equal`:
   $$\frac{(y - p)^2}{p} = 0 \iff y = p \quad \text{for } p > 0$$

---

## 5. Deep Empirical & Monte Carlo Simulation Gate

Execute the Phase 4 test suite in `tests/test_loss.py`:

| Evaluation Dimension | Experimental Protocol | Success Criterion / Bound |
| :--- | :--- | :--- |
| **Monte Carlo Label Noise Stress Test** | $10^5$ trials under symmetric label noise $\epsilon \in [0.0, 0.3]$ | Gradient variance $\operatorname{Var}(\nabla \mathcal{L}_{1/8}) \le 0.50 \times \operatorname{Var}(\nabla \mathcal{L}_{\text{CE}})$ |
| **Simplex Boundary Stability** | Evaluate gradient across $p_k \in [10^{-9}, 1 - 10^{-9}]$ | Zero NaNs, zero Infs, bounded gradient |
| **Fisher Information Ratio** | Hessian ratio $H(D_A) / H(D_{\text{KL}})$ at $\mathbf{p} = \mathbf{y}$ | Exactly $[2.0, 2.0, \dots, 2.0]$ |
| **Strict Propriety & Monotonicity** | Verify $\frac{\partial \mathcal{L}_{1/8}}{\partial p_k} < 0$ across $10^5$ samples | Monotonically decreasing on $(0, 1]$ |
| **Minimum Value** | Evaluate $\min_{p_k \in (0, 1]} \mathcal{L}_{1/8}(p_k)$ | Exactly $0.000000$ at $p_k = 1.0$ |
| **OACE vs. Cross-Entropy Benchmark** | Training convergence rate and gradient variance on classification benchmarks vs. standard Cross-Entropy ($-\ln p$) | Final convergence loss at par or slightly down ($\le 5\%$ margin vs. Cross-Entropy); gradient variance strictly lower under label noise |
| **Zero Transcendental Audit** | Grep of loss module for `log`, `ln`, `cross_entropy` | Exactly $0$ occurrences |

---

## 6. Adaptive Failure-Repair & Bidirectional Dependency Protocol

When a test or gate fails in Phase 4:
1. **Iterate Locally:** Calibrate constant scale factor $\gamma \approx 2.0$ to align the initial gradient magnitude with standard cross-entropy step size.
2. **Backward Rollback to Phase 2:** If probabilities reach exact zero during evaluation causing numerical faults, verify the A-Softmax attention sink $\Omega$. If $\Omega$ must be increased or modified, backtrack to Phase 2, adjust `src/attention.py`, re-run Phase 2 gates, and cascade forward to Phase 4.
3. **Forward Dependency Cascading:**
   - **Phases 7, 8, 9 (`src/model.py`, `scripts/run_pilot_15m.py`, `scripts/run_pretrain_*.py`):** The training loss objective and metric loggers must be updated across all downstream pretraining pipelines to reflect any changes to $\mathcal{L}_{1/8}$ or $\gamma$.
   - Synchronize Lean 4 theorems in `formal/AlgebraicTheory/Loss.lean`.

---

## 7. PASS Gates

- [ ] `formal/AlgebraicTheory/Loss.lean` compiles with 0 errors via `/root/.elan/bin/lake build`.
- [ ] `src/loss.py` created with JAX implementation of OACE $\mathcal{L}_{1/8}$ and Pearson $\chi^2$ divergence.
- [ ] $10^5$-trial Monte Carlo label noise simulation proves $\le 50\%$ gradient variance vs. Cross-Entropy.
- [ ] Simplex boundary evaluation confirms zero gradient singularities at $p_k = 10^{-9}$.
- [ ] Fisher information equivalence ratio is identically $2.0$.
- [ ] OACE loss strictly proper, monotonic, and zero at $p_k = 1.0$.
- [ ] Direct benchmark against standard Cross-Entropy confirms optimization convergence at par or within $\le 5\%$ margin.
- [ ] Zero log calls verified in loss codebase.
- [ ] `results/phase4/PASS.md` satisfies the shared PASS record contract.
