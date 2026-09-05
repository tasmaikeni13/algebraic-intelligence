# Phase 1: Pure Algebraic Primitives & Non-Linear Gating (ALU & AVN)

## 1. Objective & Research Scope
Establish the fundamental non-linear gating and variance normalization primitives of the Algebraic Stack with zero transcendental operations:
- **Algebraic Linear Unit (ALU):** $K(x) = \frac{x}{2}\left(1 + \frac{x}{\sqrt{1 + x^2}}\right) = x \beta(x)$.
- **Algebraic Variance Normalization (AVN):** $\operatorname{AVN}(\mathbf{x}) = \mathbf{x} \cdot \operatorname{rsqrt}\left(\frac{1}{d}\|\mathbf{x}\|_2^2 + \epsilon\right)$.

Verify that ALU reproduces the non-linear expressive capacity and inflection dynamics of continuous activations (GELU/Swish) while possessing exact reflection symmetry, an analytic polynomial backward pass, and strict Lipschitz bounds.

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 The Algebraic Gate Function $\beta(u)$
With the algebraic cache variable $u \coloneqq x \cdot \operatorname{rsqrt}(1 + x^2) \in (-1, 1)$:
$$\beta(u) = \frac{1 + u}{2}$$
Forward evaluation requires only one multiply, one add, and one hardware $\operatorname{rsqrt}$ call.

### 2.2 Exact Polynomial Backward Pass
The exact derivative of ALU with respect to input $x$ is expressible strictly as a cubic polynomial in the cached variable $u$:
$$\frac{d}{dx} K(x) = \beta(u) + x \beta'(x) = \frac{1}{2}\left(1 + 2u - u^3\right)$$
**Critical Zero-Transcendental Constraint:** The backward pass must NEVER recompute square roots, divisions, or transcendentals. It evaluates the forward cache $u$ directly through Horner's polynomial rule:
$$\frac{d}{dx} K(x) = 0.5 + u \cdot (1.0 - 0.5 \cdot u^2)$$

### 2.3 Inflection Point Matching with GELU
Continuous $\operatorname{GELU}(x) = x \Phi(x)$ possesses an inflection point $K''(x) = 0$ at $x \approx -0.7588$.
For ALU:
$$K''(x) = \frac{1 - u^2}{2(1 + x^2)^{1.5}} \cdot (2 - 3u^2)$$
The non-trivial inflection point occurs at $u = -\sqrt{2/3}$, corresponding to $x = -\sqrt{2} \approx -1.4142$, ensuring smooth gradient curvature without exponential saturation.

---

## 3. Lean 4 Formal Verification Gate

The agent must verify that the following formal lemmas in `formal/AlgebraicTheory/Gate.lean` and `formal/AlgebraicTheory/Variance.lean` compile with zero errors under `lake build`:

1. `alu_reflection_symmetry`:
   $$\forall u \in \mathbb{R}, \quad \beta(u) + \beta(-u) = 1$$
2. `alu_deriv_formula`:
   $$\forall u, \quad \frac{1 + u}{2} + \frac{u(1 - u^2)}{2} = \frac{1}{2}(1 + 2u - u^3)$$
3. `bounded_avn_norm`:
   $$\forall S \geq 0, \ d > 0, \ \epsilon \geq 0, \quad S \leq d\left(\frac{S}{d} + \epsilon\right)$$
4. `avn_coupling_identity`:
   Invariance of normalized coordinates under uniform positive scalar scaling: $\operatorname{AVN}(\alpha \mathbf{x}) = \operatorname{AVN}(\mathbf{x})$ for $\alpha > 0$ when $\epsilon \to 0$.

---

## 4. Mathematical Analysis & Python Verification Gate

The agent must execute `python3 analysis/verify_algebraic_primitives.py` and enforce the following numerical passing criteria:

| Metric | Target Value | Tolerance / Bound |
| :--- | :--- | :--- |
| **Gate Reflection Symmetry Error** | $\|\beta(u) + \beta(-u) - 1.0\|_\infty$ | $\leq 1.0 \times 10^{-15}$ |
| **Backward Pass Exactness** | $\|\frac{dK}{dx}_{\text{poly}} - \frac{dK}{dx}_{\text{autograd}}\|_\infty$ | $\leq 5.0 \times 10^{-16}$ |
| **Max Jacobian / Lipschitz Bound** | $\sup_{x} |K'(x)|$ | $\leq 1.05$ (theorized: $\approx 1.0445$) |
| **AVN Output Variance** | $\operatorname{Var}(\operatorname{AVN}(\mathbf{x}))$ | $\in [0.99, 1.01]$ for $\mathbf{x} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ |
| **Zero Transcendental Audit** | Grep of code for `exp`, `log`, `sin`, `cos` | Exactly $0$ occurrences |

---

## 5. Failure Modes & Self-Correction Playbook

- **Symptom: Backward pass autograd mismatch ($> 10^{-15}$):**
  *Root Cause:* Forward cache variable $u$ precision mismatch or inaccurate polynomial coefficients.
  *Correction:* Ensure $u = x / \sqrt{1 + x^2}$ is detached and cached in FP32/FP64 during forward, and the Horner sequence `0.5 + u * (1.0 - 0.5 * u * u)` is evaluated without premature rounding.
- **Symptom: Activation explosion during deep stacking:**
  *Root Cause:* Absence of AVN pre-bounding before ALU gating.
  *Correction:* Apply AVN prior to every projection: $\mathbf{h}_{\ell+1} = \operatorname{ALU}(W_1 \operatorname{AVN}(\mathbf{h}_\ell)) \odot W_2 \operatorname{AVN}(\mathbf{h}_\ell)$.

---

## 6. Passing Gate Checklist
- [ ] `formal/AlgebraicTheory/Gate.lean` compiles with 0 errors via `lake build`.
- [ ] `formal/AlgebraicTheory/Variance.lean` compiles with 0 errors via `lake build`.
- [ ] Numerical verification script passes all bounds in Section 4.
- [ ] Codebase audit confirms zero transcendental calls.
