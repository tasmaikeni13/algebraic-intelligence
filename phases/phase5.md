# Phase 5: Factorized Curvature Optimization & Rational Scheduling (ACO & ARDS)

## 1. Objective, Scientific Hypothesis & Competing Models
Eliminate all continuous exponential moving averages ($\beta_1^t, \beta_2^t$) and transcendental cosine schedules from neural optimization:
$$\textbf{"Can factorized curvature preconditioning achieve second-order natural gradient convergence in O(d) memory?"}$$

### Competing Hypotheses:
- **$H_1$ (Algebraic Hypothesis):** Factorizing the second-moment tensor into row and column marginal projections $\hat{\mathbf{V}}_{ij} = \frac{\hat{r}_i \hat{c}_j}{\bar{r}}$ reconstructs Kronecker Fisher curvature, compresses second-moment optimizer memory from $\mathcal{O}(d_{\text{out}} \cdot d_{\text{in}})$ to $\mathcal{O}(d_{\text{out}} + d_{\text{in}})$, and achieves $\mathcal{O}(1/\sqrt{T})$ convergence when paired with the Algebraic Rational Decay Schedule (ARDS) $\eta_t = \eta_0 / \sqrt{1 + \alpha t^2}$.
- **$H_0$ (Transcendental Baseline Hypothesis):** Dense coordinate-wise second moments (AdamW) and cosine annealing are essential for adaptive learning rates; factorized marginals induce destructive cross-talk across gradient coordinates, leading to divergence on ill-conditioned loss surfaces.

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 Pure Rational Momentum
$$\mathbf{M}_t = \left(1 - \frac{1}{\tau_1}\right) \mathbf{M}_{t-1} + \frac{1}{\tau_1} \mathbf{G}_t$$

### 2.2 Factorized Curvature Preconditioner
$$\mathbf{r}_t = \left(1 - \frac{1}{\tau_2}\right) \mathbf{r}_{t-1} + \frac{1}{\tau_2} \left( \frac{1}{d_{\text{in}}} \sum_{j=1}^{d_{\text{in}}} \mathbf{G}_{t, \cdot j}^{\odot 2} \right)$$
$$\mathbf{c}_t = \left(1 - \frac{1}{\tau_2}\right) \mathbf{c}_{t-1} + \frac{1}{\tau_2} \left( \frac{1}{d_{\text{out}}} \sum_{i=1}^{d_{\text{out}}} \mathbf{G}_{ti, \cdot}^{\odot 2} \right)$$
With scalar trace normalizer $\bar{r}_t = \frac{1}{d_{\text{out}}}\sum r_{t, i}$, the update step is:
$$\mathbf{U}_{t, ij} = \frac{\hat{\mathbf{M}}_{t, ij} \cdot \sqrt{\bar{r}_t}}{\sqrt{\hat{r}_{t, i}} \cdot \sqrt{\hat{c}_{t, j}} + \epsilon \sqrt{\bar{r}_t}}$$
Memory complexity: stores only $(d_{\text{out}} + d_{\text{in}})$ floats for curvature instead of $d_{\text{out}} \cdot d_{\text{in}}$.

### 2.3 Algebraic Rational Decay Schedule (ARDS)
$$\eta_t = \eta_0 \cdot \operatorname{rsqrt}\left(1 + \alpha t^2\right) = \frac{\eta_0}{\sqrt{1 + \alpha t^2}}$$

---

## 3. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Curvature.lean` with zero errors under `lake build`:

1. `factorized_rank1_recovery`:
   $$\forall i, j, \quad \frac{(a_i \bar{b})(b_j \bar{a})}{\bar{a}\bar{b}} = a_i b_j$$
2. `debiasing_identity`:
   Exact polynomial moment debiasing $\frac{v_t}{1 - \beta^t}$.
3. `decoupled_weight_decay_step`:
   Decoupled algebraic parameter update invariance.

---

## 4. Deep Empirical & Monte Carlo Simulation Gate

The agent must execute the Phase 5 test suite in `analysis/verify_algebraic_primitives.py` and `analysis/benchmark_algebraic_vs_transcendental.py`:

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
  - *Correction:* Add algebraic diagonal damping: $\hat{\mathbf{V}}_{ij} \leftarrow \hat{\mathbf{V}}_{ij} + \epsilon \bar{r}$.
- **Symptom: Optimization oscillates under large batch sizes:**
  - *Root Cause:* Momentum horizon $\tau_1$ too small relative to batch gradient variance.
  - *Correction:* Scale rational momentum horizon $\tau_1 = \tau_0 \cdot \sqrt{B / B_{\text{ref}}}$.

---

## 6. Passing Gate Checklist
- [ ] `formal/AlgebraicTheory/Curvature.lean` compiles with 0 errors via `lake build`.
- [ ] Ill-conditioned optimization sweep confirms $> 99.99\%$ convergence across $\kappa \in [10^2, 10^6]$.
- [ ] Non-convex stochastic benchmarks match AdamW convergence within $5\%$.
- [ ] Second-moment memory compression verified to be $\ge 1024\times$ at $d=4096$.
- [ ] ARDS schedule verified strictly monotonic and rational.
- [ ] Zero exponential or cosine calls confirmed in optimizer.
