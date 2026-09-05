# Phase 4: Algebraic Loss Functionals & Information Metrics (OACE / $\mathcal{L}_{1/8}$)

## 1. Objective & Research Scope
Eliminate the natural logarithm $\ln(x)$ and Shannon entropy from neural training objectives. Formulate, formally verify, and benchmark the **Optimal Algebraic Cross-Entropy (OACE / $\mathcal{L}_{1/8}$)** and the **Algebraic Divergence (AD)**:
- Prove strict proper scoring rule behavior without transcendentals.
- Demonstrate hardware-optimal backward evaluation via 3 successive $\operatorname{rsqrt}$ operations.
- Establish that OACE eliminates logarithmic gradient explosion near simplex boundaries while preserving Fisher information curvature.

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 The $\alpha$-Algebraic Cross-Entropy Family
For probability distribution $\mathbf{p} \in \operatorname{int}\Delta^{K-1}$ and one-hot target $\mathbf{y} \in \Delta^{K-1}$, the $\alpha$-ACE family is defined for $\alpha \in (0, 1]$ as:
$$\mathcal{L}_\alpha(\mathbf{p}, \mathbf{y}) \coloneqq \sum_{i=1}^K \left[ \frac{y_i}{\alpha} \left( p_i^{-\alpha} - 1 \right) + \frac{1 - \alpha}{\alpha} \left( p_i^\alpha - 1 \right) \right]$$

### 2.2 Canonical Octo-Algebraic Cross-Entropy ($\mathcal{L}_{1/8}$)
Setting $\alpha = 1/8$ matches the octic sharpening $n = 8$ of A-Softmax. For target index $k$ ($y_k = 1$):
$$\mathcal{L}_{1/8}(p_k) = 8\left( p_k^{-1/8} - 1 \right)$$
Evaluation of $p_k^{-1/8} = \operatorname{rsqrt}_3(p_k)$ proceeds via exactly 3 hardware square-root/rsqrt operations:
$$z_1 = \operatorname{rsqrt}(p_k) = p_k^{-1/2}, \quad z_2 = \operatorname{rsqrt}(z_1^{-1}) = p_k^{-1/4}, \quad z_3 = \operatorname{rsqrt}(z_2^{-1}) = p_k^{-1/8}$$
with zero log calls.

### 2.3 Pearson $\chi^2$ Divergence Equivalence
For $\alpha = 1$, the algebraic loss reduces to the Pearson $\chi^2$ divergence:
$$\mathcal{L}_1(\mathbf{p}, \mathbf{y}) = \sum_{i=1}^K \frac{(y_i - p_i)^2}{p_i} = \sum_{i=1}^K \left( \frac{y_i^2}{p_i} - 2y_i + p_i \right)$$
which provides a strictly positive, convex Riemannian metric on the probability simplex.

---

## 3. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Loss.lean` with zero errors under `lake build`:

1. `pearson_chi_sq_expansion`:
   $$\forall y, p \in \mathbb{R}, \ p \neq 0 \implies \frac{(y - p)^2}{p} = \frac{y^2}{p} - 2y + p$$
2. `pearson_divergence_nonneg`:
   $$\forall y, p \in \mathbb{R}, \ p > 0 \implies \frac{(y - p)^2}{p} \geq 0$$
3. `pearson_zero_iff_equal`:
   $$\frac{(y - p)^2}{p} = 0 \iff y = p \quad \text{for } p > 0$$

---

## 4. Mathematical Analysis & Python Verification Gate

The agent must execute the loss verification in `analysis/verify_algebraic_primitives.py`:

| Metric | Target Value | Tolerance / Bound |
| :--- | :--- | :--- |
| **Monotonicity Check** | $\frac{\partial \mathcal{L}_{1/8}}{\partial p_k} < 0$ for $p_k \in (0, 1)$ | Verified across $10^5$ samples |
| **Gradient Magnitude at $p_k = 0.01$** | $|\nabla_{p_k} \mathcal{L}_{1/8}|$ | $\approx 1.77 \times 10^2$ (vs. Cross-Entropy: $1.00 \times 10^2$) |
| **Minimum Value** | $\min_{p_k \in (0, 1]} \mathcal{L}_{1/8}(p_k)$ | Exactly $0.0$ at $p_k = 1.0$ |
| **Zero Transcendental Audit** | Grep of loss implementation for `log`, `log_softmax` | Exactly $0$ occurrences |

---

## 5. Failure Modes & Self-Correction Playbook

- **Symptom: Gradient overflow when $p_k \to 0$:**
  *Root Cause:* Zero probability prediction causes division by zero in $p_k^{-1/8}$.
  *Correction:* The A-Softmax attention sink $\Omega > 0$ and AVN pre-bounding mathematically guarantee a non-zero lower bound:
  $$p_{\min} \geq \frac{\kappa_8(-B)}{K \kappa_8(B) + \Omega} > 0$$
  If standalone logits are passed, clamp probability input at $p_{\text{floor}} = 10^{-7}$ using purely algebraic clamping.
- **Symptom: Training loss plateaus too early:**
  *Root Cause:* Scaling factor in $\mathcal{L}_{1/8}$ too small relative to learning rate.
  *Correction:* Scale the loss by a constant factor $\gamma = 2.0$ to calibrate gradient step norms against standard cross-entropy.

---

## 6. Passing Gate Checklist
- [ ] `formal/AlgebraicTheory/Loss.lean` compiles with 0 errors via `lake build`.
- [ ] OACE monotonicity and non-negativity verified numerically across $(0, 1]$.
- [ ] Three-step $\operatorname{rsqrt}$ implementation verified against float64 reference.
- [ ] Zero log calls verified in loss module.
