# Phase 3: Algebraic Geometric Oscillators & Shift Equivariance (AGO)

Start only after Phase 2 PASS. Read `theory.md`, `formal/README.md`, `formal/AlgebraicTheory/Cayley.lean`, Phase 2 evidence in `results/phase2/`, and `phases/README.md` completely before executing. Execute the shared adaptive failure-repair loop until all gates pass.

---

## 1. Objective, Scientific Hypothesis & Competing Models

Eliminate all trigonometric functions ($\sin, \cos$) and continuous complex exponentials from Transformer positional encodings:
$$\textbf{"Can rational Cayley rotations provide exact shift equivariance and long-context length generalization?"}$$

### Competing Hypotheses:
- **$H_1$ (Algebraic Hypothesis):** The birational Cayley transform on $\mathfrak{so}(2)$ yields purely rational orthogonal matrices $\mathbf{R}(w_k) = \frac{1}{1 + w_k^2}\begin{pmatrix} 1 - w_k^2 & -2w_k \\ 2w_k & 1 - w_k^2 \end{pmatrix}$ that conserve Euclidean norms $\|\mathbf{R}(w)\mathbf{v}\| = \|\mathbf{v}\|$, maintain exact group closure with $\det = 1$, provide strict relative shift equivariance $\langle\mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k}\rangle = f(n - m)$, and permit $\mathcal{O}(1)$ autoregressive recurrence without trigonometric series.
- **$H_0$ (Transcendental Baseline Hypothesis):** Transcendental harmonic frequencies (sinusoids / RoPE) are indispensable for rotary positional encoding; rational mappings will suffer from cumulative numerical drift, frequency aliasing, or failure on out-of-distribution sequence lengths.

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 The Birational Cayley Transform on $\mathfrak{so}(2)$
For channel pair $k \in \{0, \dots, d/2 - 1\}$, define the rational frequency parameter $w_k \in (0, 1]$:
$$\mathbf{R}(w_k) = (\mathbf{I} - w_k \mathbf{J})(\mathbf{I} + w_k \mathbf{J})^{-1} = \frac{1}{1 + w_k^2} \begin{pmatrix} 1 - w_k^2 & -2w_k \\ 2w_k & 1 - w_k^2 \end{pmatrix}$$
where $\mathbf{J} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$.

### 2.2 Invariant Properties
1. **Unimodularity:** $\det(\mathbf{R}(w_k)) = \frac{(1 - w_k^2)^2 + (2w_k)^2}{(1 + w_k^2)^2} = 1$ strictly.
2. **Norm Conservation:** $\|\mathbf{R}(w_k) \mathbf{v}\|_2 = \|\mathbf{v}\|_2$ for all $\mathbf{v} \in \mathbb{R}^2$.
3. **Shift Equivariance:** $\mathbf{R}(w_k)^m \cdot \mathbf{R}(w_k)^n = \mathbf{R}(w_k)^{m+n}$ and $(\mathbf{R}(w_k)^m)^\top \mathbf{R}(w_k)^n = \mathbf{R}(w_k)^{n-m}$.
4. **$\mathcal{O}(1)$ Autoregressive Update:** $\mathbf{R}_k(m) = \mathbf{R}_k \mathbf{R}_k(m-1)$ computed in 4 FMAs per channel pair on the TPU v4 VMU with periodic algebraic re-normalization $\mathbf{v} \leftarrow \mathbf{v} \cdot \operatorname{rsqrt}(\|\mathbf{v}\|^2)$.

---

## 3. Implementation Target: JAX / TPU v4 Architecture

Instruct the creation and verification of the following components targeting the 16 TPU v4 Pod:
1. **`src/attention.py` (AGO module additions)**:
   - `build_cayley_rotary_matrix(dim, max_seq_len)`: Precomputes rational Cayley rotation parameters without `sin` or `cos`.
   - `apply_ago_rotations(q, k)`: JAX vectorized rotary application executing natively across TPU v4 VMU vector registers.
2. **`tests/test_ago.py`**:
   - Verification of unimodularity ($\det = 1$), norm preservation across long sequences ($L \ge 8192$), and shift equivariance.
   - AST audit confirming zero occurrences of `sin`, `cos`, or complex numbers.

---

## 4. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Cayley.lean` with zero errors under `/root/.elan/bin/lake build`:

1. `cayley_pythagorean_identity`:
   $$\forall w \in \mathbb{R}, \quad (1 - w^2)^2 + (2w)^2 = (1 + w^2)^2$$
2. `cayley_col1_norm_sq` & `cayley_col2_norm_sq`:
   Exact unit norm preservation for columns: $((1-w^2)/(1+w^2))^2 + (2w/(1+w^2))^2 = 1$.
3. `cayley_dot_product_zero`:
   Orthogonality of rotation column vectors $\mathbf{c}_1 \cdot \mathbf{c}_2 = 0$.
4. `cayley_det_one`:
   Strict unimodularity $\det(\mathbf{R}(w)) = 1$, certifying $\mathrm{SO}(2)$ Lie group membership.
5. `cayley_norm_preserving`:
   Invariance of 2D Euclidean norm: $\|\mathbf{R}(w) \mathbf{v}\|_2 = \|\mathbf{v}\|_2$.

---

## 5. Deep Empirical & Monte Carlo Simulation Gate

Execute the Phase 3 verification suite in `tests/test_ago.py`:

| Evaluation Dimension | Experimental Protocol | Success Criterion / Bound |
| :--- | :--- | :--- |
| **Long-Context Shift Equivariance** | All position pairs $(m, n) \le 4096$, measure $\|\mathbf{R}_m^\top \mathbf{R}_n - \mathbf{R}_{n-m}\|_\infty$ | $\leq 1.0 \times 10^{-6}$ |
| **Relative Attention Dot Product Error** | $10^5$ random query/key pairs across context $L \in [128, 4096]$ | $\leq 1.0 \times 10^{-6}$ |
| **Cumulative Norm Conservation Drift** | Sequence lengths $m \in [1, 8192]$, measure $|\|\mathbf{R}^m \mathbf{v}\|_2 - \|\mathbf{v}\|_2|$ | $\leq 1.0 \times 10^{-6}$ |
| **Cayley Determinant Error** | $10^5$ frequency samples $w \in [10^{-5}, 10^2]$, measure $|\det(\mathbf{R}(w)) - 1.0|$ | $\leq 1.0 \times 10^{-15}$ |
| **Column Orthogonality Error** | Inner product $|\mathbf{c}_1 \cdot \mathbf{c}_2|$ across $10^5$ samples | $\leq 1.0 \times 10^{-15}$ |
| **Associative Recall on Out-of-Dist Context** | Train on $L=256$, test on $L=1024$ and $L=2048$ | Retrieval accuracy $\ge 95.0\%$ |
| **AGO vs. RoPE Benchmark** | Direct head-to-head microbenchmark of AGO vs. standard trigonometric RoPE on rotary embedding latency and associative recall | Throughput $\ge 90\%$ of RoPE; associative recall retrieval at par or slightly down ($\le 3\%$ absolute accuracy delta) |
| **Zero Trigonometric Audit** | Grep of AGO module for `sin`, `cos` | Exactly $0$ occurrences |

---

## 6. Adaptive Failure-Repair & Bidirectional Dependency Protocol

When a test or gate fails in Phase 3:
1. **Iterate Locally:** If cumulative precision roundoff occurs at $L > 2048$, apply algebraic re-normalization using $\operatorname{rsqrt}(c^2 + s^2)$.
2. **Backward Rollback to Phase 1:** If re-normalization requires specific $\operatorname{rsqrt}$ precision or numerical damping, ensure consistency with Phase 1 AVN radical operators.
3. **Forward Dependency Cascading:**
   - **Phase 6 (`src/kernels/pallas_afa.py`):** If Cayley matrix representation changes (e.g. 2D planar rotation packing vs complex representation), update the Pallas TPU kernel tile loading logic accordingly.
   - **Phases 7, 8, 9 (`src/model.py`):** Update positional encoding application in `AlgebraicTransformerLM`.
   - Re-verify that `formal/AlgebraicTheory/Cayley.lean` and `lake build` remain completely aligned.

---

## 7. PASS Gates

- [ ] `formal/AlgebraicTheory/Cayley.lean` compiles with 0 errors via `/root/.elan/bin/lake build`.
- [ ] JAX implementation of AGO Cayley rotary embeddings created in `src/attention.py`.
- [ ] Matrix shift equivariance error $\le 1.0 \times 10^{-6}$ across context length 4096.
- [ ] Cumulative rotation norm drift $\le 1.0 \times 10^{-6}$ up to $m = 8192$.
- [ ] Determinant is verified to be identically $1.0$ (error $\le 1.0 \times 10^{-15}$).
- [ ] Column orthogonality error $\le 1.0 \times 10^{-15}$.
- [ ] Out-of-distribution associative recall reaches $\ge 95\%$ accuracy.
- [ ] Head-to-head benchmark against standard trigonometric RoPE confirms throughput $\ge 90\%$ and retrieval accuracy at par or within $\le 3\%$ margin.
- [ ] Zero trigonometric calls verified in code.
- [ ] `results/phase3/PASS.md` satisfies the shared PASS record contract.
