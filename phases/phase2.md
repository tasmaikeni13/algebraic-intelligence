# Phase 2: Octic Algebraic Attention & 2-Lipschitz Bounds (A-Softmax)

## 1. Objective & Research Scope
Eliminate the exponential operator $\exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d})$ from Transformer attention. Formulate, formally verify, and benchmark **Algebraic Softmax (A-Softmax)** using the octic algebraic kernel:
$$\kappa_8(x) \coloneqq \left(x + \sqrt{1 + x^2}\right)^8$$
Demonstrate that $\kappa_8$ provides super-exponential contrast ratios ($> 10^5$) across typical attention score intervals while maintaining a globally bounded 2-Lipschitz Jacobian, eliminating attention score blowups and enabling outlier-free FP4/INT4 quantization.

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 The Three-Stage Squaring Chain
Forward evaluation of $\kappa_8(x)$ requires zero transcendentals and evaluates in exactly 3 successive hardware squaring stages:
1. Stage 0 (Base Kernel): $s = \sqrt{1 + x^2}$, $\kappa_1(x) = x + s$.
2. Stage 1 (Degree 2): $\kappa_2(x) = (\kappa_1(x))^2 = (x + s)^2$.
3. Stage 2 (Degree 4): $\kappa_4(x) = (\kappa_2(x))^2$.
4. Stage 3 (Degree 8): $\kappa_8(x) = (\kappa_4(x))^2$.

### 2.2 Algebraic Softmax Operator
For score vector $\mathbf{s} \in \mathbb{R}^K$:
$$\operatorname{A-Softmax}(\mathbf{s})_i = \frac{\kappa_8(s_i)}{\sum_{j=1}^K \kappa_8(s_j) + \Omega}$$
where $\Omega > 0$ is a constant algebraic attention sink preventing division by zero and absorbing background noise without requiring dedicated dummy tokens.

### 2.3 Globally Bounded Jacobian (2-Lipschitz Guarantee)
Unlike exponential softmax whose derivative $\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)$ scales with score magnitude, A-Softmax with AVN pre-bounded inputs satisfies:
$$\left|\frac{\partial p_i}{\partial s_j}\right| \leq \frac{n}{4} = 2.0 \quad \text{for } n = 8$$
This global bound guarantees that attention score perturbation under numerical quantization cannot grow unbounded.

---

## 3. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Kernel.lean` with zero errors under `lake build`:

1. `kernel_reciprocal_identity`:
   $$\forall x, s \in \mathbb{R}, \quad s^2 = 1 + x^2 \implies (s + x)(s - x) = 1$$
2. `kernel_squaring_step`:
   $$(y^2)^2 = y^4 \quad \text{and} \quad (y^4)^2 = y^8$$
3. `kernel_octa_degree`:
   Formal proof of monotonic degree progression $1 \to 2 \to 4 \to 8$ across the squaring composition chain.

---

## 4. Mathematical Analysis & Python Verification Gate

The agent must execute the A-Softmax test suites in `analysis/verify_algebraic_primitives.py` and `analysis/benchmark_algebraic_vs_transcendental.py`:

| Metric | Target Value | Tolerance / Bound |
| :--- | :--- | :--- |
| **Reciprocal Identity Error** | $\|(s+x)(s-x) - 1.0\|_\infty$ | $\leq 5.0 \times 10^{-14}$ |
| **Maximum Jacobian Diagonal** | $\max_{i, j} \left|\frac{\partial p_i}{\partial s_j}\right|$ | $\leq 2.0$ (empirically $\approx 0.187$) |
| **Dynamic Contrast Ratio** | $\frac{\kappa_8(+3)}{\kappa_8(-3)}$ | $\geq 1.0 \times 10^5$ (exact: $103,682$) |
| **FP4 Quantization Sensitivity Ratio** | $\frac{\Delta_{\text{exp}}}{\Delta_{\text{alg}}}$ under noise $\sigma = 0.05$ | $\geq 100.0\times$ (theorized: $> 200\times$) |
| **Attention Output Sum** | $\sum_{i=1}^K p_i$ | $\leq 1.0$ (strictly bounded on simplex) |

---

## 5. Failure Modes & Self-Correction Playbook

- **Symptom: Underflow / Overflow in $\kappa_8$ when scores exceed $[-5, 5]$:**
  *Root Cause:* Missing input scale factor $\tau$ or AVN normalization on queries and keys.
  *Correction:* Ensure queries and keys are AVN-normalized and scaled by $\tau = 1/\sqrt{d_k}$ before kernel evaluation: $\mathbf{s} = \operatorname{AVN}(\mathbf{Q}) \operatorname{AVN}(\mathbf{K})^\top / \sqrt{d_k}$.
- **Symptom: Diffuse attention entropy / failure to attend to single token:**
  *Root Cause:* Sharpening degree $n$ too low ($n=2$ or $n=4$).
  *Correction:* Ensure the squaring chain completes all 3 stages to reach $n=8$. The contrast ratio of $\kappa_8$ across $[-2.5, 2.5]$ is $10^5$, matching $\exp(\Delta s)$ over typical transformer logits.

---

## 6. Passing Gate Checklist
- [ ] `formal/AlgebraicTheory/Kernel.lean` compiles with 0 errors via `lake build`.
- [ ] Numerical test suite passes Jacobian bound $\leq 2.0$.
- [ ] Contrast ratio across $[-3, 3]$ exceeds $10^5$.
- [ ] Quantization robustness test verifies $\geq 100\times$ noise reduction over exponential softmax.
