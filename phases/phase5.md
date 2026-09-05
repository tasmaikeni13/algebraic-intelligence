# Phase 5: Factorized Curvature Optimization & Rational Scheduling (ACO & ARDS)

## 1. Objective & Research Scope
Eliminate all continuous exponential moving averages ($\beta_1^t, \beta_2^t$) and transcendental cosine annealing schedules from neural optimization. Formulate, formally verify, and benchmark the **Algebraic Curvature Optimizer (ACO)** and the **Algebraic Rational Decay Schedule (ARDS)**:
- Achieve second-order natural gradient curvature preconditioning while compressing second-moment optimizer memory from $\mathcal{O}(d_{\text{out}} \cdot d_{\text{in}})$ to $\mathcal{O}(d_{\text{out}} + d_{\text{in}})$ in HBM.
- Establish convergence rates of $\mathcal{O}(1/\sqrt{T})$ on non-convex landscapes using rational learning rate schedules.

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 Pure Rational Momentum
For gradient matrix $\mathbf{G}_t \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and rational momentum horizon $\tau_1 \in \mathbb{N}$:
$$\mathbf{M}_t = \left(1 - \frac{1}{\tau_1}\right) \mathbf{M}_{t-1} + \frac{1}{\tau_1} \mathbf{G}_t$$

### 2.2 Factorized Curvature Preconditioner
Instead of storing the dense $d_{\text{out}} \times d_{\text{in}}$ matrix of squared gradients, accumulate only the row and column marginal projections:
$$\mathbf{r}_t = \left(1 - \frac{1}{\tau_2}\right) \mathbf{r}_{t-1} + \frac{1}{\tau_2} \left( \frac{1}{d_{\text{in}}} \sum_{j=1}^{d_{\text{in}}} \mathbf{G}_{t, \cdot j}^{\odot 2} \right)$$
$$\mathbf{c}_t = \left(1 - \frac{1}{\tau_2}\right) \mathbf{c}_{t-1} + \frac{1}{\tau_2} \left( \frac{1}{d_{\text{out}}} \sum_{i=1}^{d_{\text{out}}} \mathbf{G}_{ti, \cdot}^{\odot 2} \right)$$
With scalar trace mean $\bar{r}_t = \frac{1}{d_{\text{out}}}\sum_{i=1}^{d_{\text{out}}} r_{t, i}$, the reconstructed second-moment curvature tensor is:
$$\hat{\mathbf{V}}_{t, ij} = \frac{\hat{r}_{t, i} \cdot \hat{c}_{t, j}}{\bar{r}_t}$$
Preconditioned gradient steps are evaluated on-the-fly without materializing $\hat{\mathbf{V}}$ in HBM:
$$\mathbf{U}_{t, ij} = \frac{\hat{\mathbf{M}}_{t, ij}}{\sqrt{\hat{r}_{t, i} \hat{c}_{t, j} / \bar{r}_t} + \epsilon} = \frac{\hat{\mathbf{M}}_{t, ij} \cdot \sqrt{\bar{r}_t}}{\sqrt{\hat{r}_{t, i}} \cdot \sqrt{\hat{c}_{t, j}} + \epsilon \sqrt{\bar{r}_t}}$$

### 2.3 Algebraic Rational Decay Schedule (ARDS)
Replace transcendental cosine decay $\eta_t = \frac{\eta_0}{2}(1 + \cos(\pi t / T))$ with the rational decay schedule:
$$\eta_t = \eta_0 \cdot \operatorname{rsqrt}\left(1 + \alpha t^2\right) = \frac{\eta_0}{\sqrt{1 + \alpha t^2}}$$
where $\alpha = \frac{(\eta_0 / \eta_{\min})^2 - 1}{T^2}$, ensuring exact boundary matching $\eta_T = \eta_{\min}$ with zero transcendental evaluations.

---

## 3. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Curvature.lean` with zero errors under `lake build`:

1. `factorized_rank1_recovery`:
   For any rank-1 curvature matrix $\mathbf{V} = \mathbf{a} \mathbf{b}^\top$ with $\bar{a} = \frac{1}{m}\sum a_i$ and $\bar{b} = \frac{1}{n}\sum b_j$:
   $$\forall i, j, \quad \frac{(a_i \bar{b})(b_j \bar{a})}{\bar{a}\bar{b}} = a_i b_j$$
2. `debiasing_identity`:
   Exact recovery of un-biased polynomial moment $\frac{v_t}{1 - \beta^t}$.
3. `decoupled_weight_decay_step`:
   Decoupled algebraic parameter update invariance: $W_{t+1} = W_t(1 - \lambda \eta_t) - \eta_t \mathbf{P}_t^{-1} \mathbf{M}_t$.

---

## 4. Mathematical Analysis & Python Verification Gate

The agent must execute `analysis/verify_algebraic_primitives.py` and `analysis/benchmark_algebraic_vs_transcendental.py`:

| Metric | Target Value | Tolerance / Bound |
| :--- | :--- | :--- |
| **Ill-Conditioned Quadratic Loss** | $\frac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x}$ ($\kappa = 1000$) | Reaches $\leq 1.0 \times 10^{-6}$ in 200 steps |
| **Memory Compression Ratio** | $\frac{\text{Memory}(\text{AdamW})}{\text{Memory}(\text{ACO})}$ at $4096 \times 4096$ | $\geq 2.0\times$ (Total), $\geq 2048\times$ (Second Moment) |
| **Memory Compression Ratio** | at $8192 \times 8192$ | $\geq 4096\times$ (Second Moment) |
| **ARDS Schedule Range** | $\eta_0 \to \eta_T$ | Exactly monotonically decreasing |
| **Zero Transcendental Audit** | Grep of ACO for `exp`, `log`, `cos` | Exactly $0$ occurrences |

---

## 5. Failure Modes & Self-Correction Playbook

- **Symptom: Optimization oscillates or diverges on anisotropic gradients:**
  *Root Cause:* Preconditioner units mismatched due to omitting the scalar trace normalizer $\bar{r}_t$.
  *Correction:* Ensure the normalization factor $\bar{r} = \operatorname{mean}(\mathbf{r})$ is included so that $\frac{r_i c_j}{\bar{r}}$ has units $[units]^2$, matching the gradient square dimension.
- **Symptom: Momentum vanishes on long training horizons:**
  *Root Cause:* Numerical underflow in continuous accumulation $(1 - 1/\tau)^t$.
  *Correction:* Use rational integer accumulation with polynomial debiasing $1 - (1 - 1/\tau_1)^t$.

---

## 6. Passing Gate Checklist
- [ ] `formal/AlgebraicTheory/Curvature.lean` compiles with 0 errors via `lake build`.
- [ ] Quadratic benchmark with $\kappa = 1000$ converges from $> 80,000$ to $0.000000$.
- [ ] Memory footprint verifies $\mathcal{O}(d_{\text{out}} + d_{\text{in}})$ scaling.
- [ ] Zero exponential or cosine schedule calls in optimizer.
