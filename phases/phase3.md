# Phase 3: Algebraic Geometric Oscillators & Shift Equivariance (AGO)

## 1. Objective & Research Scope
Eliminate all trigonometric functions ($\sin, \cos$) from Transformer positional representations. Formulate, formally verify, and benchmark **Algebraic Geometric Oscillators (AGO)**:
- Generate exact orthogonal 2D rotation matrices in $\mathrm{SO}(2)$ using the **rational Cayley transform** of a static skew-symmetric generator $\mathbf{A} \in \mathfrak{so}(2)$.
- Prove strict shift equivariance: $\langle \mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k} \rangle = f(\mathbf{q}, \mathbf{k}, n - m)$ without transcendental series.
- Demonstrate $\mathcal{O}(1)$ autoregressive token generation via cached matrix-vector recurrence.

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 The Birational Cayley Transform on $\mathfrak{so}(2)$
For channel pair $k \in \{0, \dots, d/2 - 1\}$, define the rational frequency parameter $w_k \in \mathbb{Q}^+$. The Cayley transform yields the purely rational rotation matrix:
$$\mathbf{R}(w_k) = (\mathbf{I} - w_k \mathbf{J})(\mathbf{I} + w_k \mathbf{J})^{-1} = \frac{1}{1 + w_k^2} \begin{pmatrix} 1 - w_k^2 & -2w_k \\ 2w_k & 1 - w_k^2 \end{pmatrix}$$
where $\mathbf{J} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$.

### 2.2 Shift Equivariance & Orthogonality
Because $\mathbf{R}(w_k)$ is a strictly orthogonal matrix in $\mathrm{SO}(2)$:
1. Unimodularity: $\det(\mathbf{R}(w_k)) = \frac{(1 - w_k^2)^2 + (2w_k)^2}{(1 + w_k^2)^2} = \frac{1 + 2w_k^2 + w_k^4}{(1 + w_k^2)^2} = 1$.
2. Norm Invariance: $\|\mathbf{R}(w_k) \mathbf{v}\|_2 = \|\mathbf{v}\|_2$ for all $\mathbf{v} \in \mathbb{R}^2$.
3. Gram Product Identity:
   $$\mathbf{R}(w_k)^m \cdot \mathbf{R}(w_k)^n = \mathbf{R}(w_k)^{m+n}, \quad (\mathbf{R}(w_k)^m)^\top \mathbf{R}(w_k)^n = \mathbf{R}(w_k)^{n-m}$$
Thus, attention scores depend strictly on relative token distance $(n - m)$.

---

## 3. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Cayley.lean` with zero errors under `lake build`:

1. `cayley_pythagorean_identity`:
   $$\forall w \in \mathbb{R}, \quad (1 - w^2)^2 + (2w)^2 = (1 + w^2)^2$$
2. `cayley_col1_norm_sq` & `cayley_col2_norm_sq`:
   Columns of $\mathbf{R}(w)$ have exact Euclidean norm $1$:
   $$\left(\frac{1 - w^2}{1 + w^2}\right)^2 + \left(\frac{2w}{1 + w^2}\right)^2 = 1$$
3. `cayley_dot_product_zero`:
   Orthogonality of column vectors $\mathbf{c}_1 \cdot \mathbf{c}_2 = 0$.
4. `cayley_det_one`:
   $$\det(\mathbf{R}(w)) = 1 \quad (\text{exact } \mathrm{SO}(2) \text{ membership})$$
5. `cayley_norm_preserving`:
   Invariance of 2D Euclidean norm: $\|\mathbf{R}(w) \mathbf{v}\|^2 = \|\mathbf{v}\|^2$.

---

## 4. Mathematical Analysis & Python Verification Gate

The agent must execute the AGO verification in `analysis/verify_algebraic_primitives.py`:

| Metric | Target Value | Tolerance / Bound |
| :--- | :--- | :--- |
| **Cayley Determinant Error** | $|\det(\mathbf{R}(w)) - 1.0|$ | $\leq 1.0 \times 10^{-15}$ |
| **Column Orthogonality Error** | $|\mathbf{c}_1 \cdot \mathbf{c}_2|$ | $\leq 1.0 \times 10^{-15}$ |
| **Shift Equivariance Error** | $\|\mathbf{R}_m^\top \mathbf{R}_n - \mathbf{R}_{n-m}\|_\infty$ | $\leq 1.0 \times 10^{-6}$ |
| **Norm Conservation Error** | $\left|\|\mathbf{R}(w)\mathbf{v}\|_2 - \|\mathbf{v}\|_2\right|$ | $\leq 1.0 \times 10^{-7}$ |
| **Zero Transcendental Audit** | Grep of AGO implementation for `sin`, `cos` | Exactly $0$ occurrences |

---

## 5. Failure Modes & Self-Correction Playbook

- **Symptom: Numerical drift in sequence positions $m > 1000$:**
  *Root Cause:* Repeated sequential multiplication of float32 rotation matrices accumulates rounding error $\epsilon \cdot m$.
  *Correction:* Precompute powers $\mathbf{R}^m$ in float64 using binary exponentiation, or parameterize positions via the rational addition formula on angles:
  $$w_{m+1} = \frac{w_m + w_1}{1 - w_m w_1}$$
  and evaluate $\mathbf{R}(w_m)$ directly.
- **Symptom: High frequency channel aliasing:**
  *Root Cause:* Choice of $w_k$ frequencies growing too large ($w \to \infty$).
  *Correction:* Bound the rational frequency spectrum by setting $w_k = \frac{1}{\beta^{2k/d}}$ where $\beta$ is a rational base (e.g., $10000$), ensuring $w_k \in (0, 1]$.

---

## 6. Passing Gate Checklist
- [x] `formal/AlgebraicTheory/Cayley.lean` compiles with 0 errors via `lake build`.
- [x] Shift equivariance error is bounded below $10^{-6}$:
  - Matrix equivariance error $\|\mathbf{R}_m^\top \mathbf{R}_n - \mathbf{R}_{n-m}\|_\infty = 5.96 \times 10^{-8} \leq 1.0 \times 10^{-6}$ [PASSED]
  - Relative attention dot product error: $4.77 \times 10^{-7} \leq 1.0 \times 10^{-6}$ [PASSED]
- [x] Determinant is verified to be identically $1.0$:
  - Determinant error $|\det(\mathbf{R}(w)) - 1.0| = 4.44 \times 10^{-16} \leq 1.0 \times 10^{-15}$ [PASSED]
  - Column orthogonality error $|\mathbf{c}_1 \cdot \mathbf{c}_2| = 0.00 \times 10^0 \leq 1.0 \times 10^{-15}$ [PASSED]
  - Norm conservation error $|\|\mathbf{R}(w)\mathbf{v}\|_2 - \|\mathbf{v}\|_2| = 8.88 \times 10^{-16} \leq 1.0 \times 10^{-7}$ [PASSED]
- [x] Zero trigonometric calls verified in code (AST & regex: 0 occurrences of `sin`, `cos`).
