# Phase 5: Factorized Curvature Optimization & Rational Scheduling (ACO & ARDS)

Start only after Phase 4 PASS. Read `theory.md`, `formal/README.md`, `formal/AlgebraicTheory/Curvature.lean`, Phase 4 evidence in `results/phase4/`, and `phases/AUTONOMY_PROTOCOL.md` completely before executing. Execute the shared failure-repair loop until all gates pass.

---

## 1. Objective, Scientific Hypothesis & Competing Models

Eliminate all continuous exponential moving averages ($\beta_1^t, \beta_2^t$) and transcendental cosine schedules from neural optimization:
$$\textbf{"Can factorized curvature preconditioning achieve second-order natural gradient convergence in O(d) memory?"}$$

### Competing Hypotheses:
- **$H_1$ (Algebraic Hypothesis):** Factorizing the second-moment tensor into row and column marginal projections $\hat{\mathbf{V}}_{ij} = \sqrt{\hat{r}_i \hat{c}_j}$ reconstructs Kronecker Fisher curvature, compresses second-moment optimizer memory from $\mathcal{O}(d_{\text{out}} \cdot d_{\text{in}})$ to $\mathcal{O}(d_{\text{out}} + d_{\text{in}})$, and achieves $\mathcal{O}(1/\sqrt{T})$ convergence when paired with the Algebraic Rational Decay Schedule (ARDS) $\eta_t \propto \operatorname{rsqrt}(1 + \alpha t^2)$.
- **$H_0$ (Transcendental Baseline Hypothesis):** Dense coordinate-wise second moments (AdamW) and cosine annealing are essential for adaptive learning rates; factorized marginals induce destructive cross-talk across gradient coordinates, leading to divergence on ill-conditioned loss surfaces.

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 Pure Rational Momentum & Debiasing
$$\mathbf{M}_t = \beta_1 \mathbf{M}_{t-1} + (1 - \beta_1) \mathbf{G}_t, \qquad \hat{\mathbf{M}}_t = \frac{\mathbf{M}_t}{1 - \beta_1^t}$$
where $\beta_1 \in \mathbb{Q}$ (e.g. $9/10$) and $1 - \beta_1^t$ is evaluated as an exact polynomial in $\mathcal{O}(\log t)$ multiplications.

### 2.2 Factorized Curvature Accumulators
For weight gradient $\mathbf{G}_t \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:
$$\mathbf{r}_t = \beta_2 \mathbf{r}_{t-1} + (1 - \beta_2) \left( \frac{1}{d_{\text{in}}} \sum_{j=1}^{d_{\text{in}}} \mathbf{G}_{t, \cdot, j}^{\odot 2} \right) \in \mathbb{R}^{d_{\text{out}}}$$
$$\mathbf{c}_t = \beta_2 \mathbf{c}_{t-1} + (1 - \beta_2) \left( \frac{1}{d_{\text{out}}} \sum_{i=1}^{d_{\text{out}}} \mathbf{G}_{t, i, \cdot}^{\odot 2} \right) \in \mathbb{R}^{d_{\text{in}}}$$
With debiased marginals $\hat{\mathbf{r}}_t = \mathbf{r}_t / (1 - \beta_2^t)$ and $\hat{\mathbf{c}}_t = \mathbf{c}_t / (1 - \beta_2^t)$, the update step evaluates on-the-fly via a single $\operatorname{rsqrt}$:
$$\mathbf{U}_{t, ij} = \hat{\mathbf{M}}_{t, ij} \cdot \operatorname{rsqrt}\left(\hat{r}_{t, i} \hat{c}_{t, j} + \epsilon^2\right)$$
Parameter update with decoupled algebraic weight decay $\lambda \in \mathbb{Q}$:
$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t \mathbf{U}_t - \eta_t \lambda \mathbf{W}_{t-1}$$

### 2.3 Algebraic Rational Decay Schedule (ARDS)
$$\eta_t = \eta_{\max} \cdot \min\left(1, \frac{t}{T_{\text{warm}}}\right) \cdot \operatorname{rsqrt}\left(1 + \alpha \left[\frac{\max(0, t - T_{\text{warm}})}{T_{\text{decay}}}\right]^2\right)$$
Evaluated in 1 subtraction, 1 square, 1 FMA, and 1 hardware $\operatorname{rsqrt}$. Zero trigonometric functions.

---

## 3. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Curvature.lean` with zero errors under `/root/.elan/bin/lake build`:

1. `factorized_rank1_recovery`:
   $$\forall i, j, \quad \frac{(a_i \bar{b})(b_j \bar{a})}{\bar{a}\bar{b}} = a_i b_j$$
2. `debiasing_identity`:
   Exact polynomial moment debiasing $\frac{v_t}{1 - \beta^t}$.
3. `decoupled_weight_decay_step`:
   Decoupled algebraic parameter update invariance.

---

## 4. Deep Empirical & Monte Carlo Simulation Gate

Execute the Phase 5 test suite in `analysis/verify_algebraic_primitives.py` and `analysis/benchmark_algebraic_vs_transcendental.py`:

| Evaluation Dimension | Experimental Protocol | Success Criterion / Bound |
| :--- | :--- | :--- |
| **Ill-Conditioned Optimization Sweep** | $10^4$ trials on quadratic surfaces with condition number $\kappa \in [10^2, 10^6]$ | Final loss reduction $> 99.99\%$ in 300 steps |
| **Non-Convex Surface Benchmarks** | Rosenbrock & Rastrigin benchmarks with stochastic noise $\sigma = 0.5$ | Converges within $5\%$ of AdamW final loss |
| **Memory Compression at Scale** | Measure optimizer state in bytes at matrix dimensions $4096$ and $8192$ | $\ge 1024\times$ (at $4096$) and $\ge 2048\times$ (at $8192$) curvature compression |
| **ARDS Schedule Monotonicity** | Sample schedule across $t \in [0, 10^5]$ steps | Strictly monotonic decay with $\mathcal{O}(1/\sqrt{T})$ asymptotic rate |
| **Zero Transcendental Audit** | Grep of ACO codebase for `exp`, `log`, `cos` | Exactly $0$ occurrences |

---

## 5. Autonomous Failure Ledger & Self-Correction Playbook

- **Symptom: Preconditioner ill-conditioning on dormant/sparse feature columns:**
  - *Root Cause:* Near-zero marginal entries in $\mathbf{c}_j$.
  - *Correction:* Add algebraic diagonal damping: $\hat{\mathbf{V}}_{ij} \leftarrow \hat{\mathbf{V}}_{ij} + \epsilon_{\text{curv}} \bar{r} \mathbf{I}$.
- **Symptom: Optimization oscillates under large batch sizes:**
  - *Root Cause:* Momentum horizon too small relative to batch gradient variance.
  - *Correction:* Scale rational momentum horizon $\tau_1 = \tau_0 \cdot \sqrt{B / B_{\text{ref}}}$.

---

## 6. Passing Gate Checklist

- [ ] `formal/AlgebraicTheory/Curvature.lean` compiles with 0 errors via `/root/.elan/bin/lake build`.
- [ ] Ill-conditioned optimization sweep confirms $> 99.99\%$ convergence across $\kappa \in [10^2, 10^6]$.
- [ ] Non-convex stochastic benchmarks match AdamW convergence within $5\%$.
- [ ] Second-moment memory compression verified to be $\ge 1024\times$ at $d=4096$ and $\ge 2048\times$ at $d=8192$.
- [ ] ARDS schedule verified strictly monotonic and rational.
- [ ] Zero exponential or cosine calls confirmed in optimizer.
- [ ] `results/phase5/PASS.md` satisfies the shared PASS record contract.
