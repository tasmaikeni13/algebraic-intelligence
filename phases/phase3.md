# Phase 3 — Learned Representations, Associative Memory, & Sequence Induction

Start only after Phase 2 PASS. Read all prior artifacts in `results/` and `phases/AUTONOMY_PROTOCOL.md`. Apply the mandatory failure-repair loop until all gates pass.

This is the first learned-parameter phase. No large-scale natural language pretraining is permitted yet. The objective is to rigorously determine whether **pure algebraic neural networks can learn in-context representations, associative memory, and sequence induction** as effectively as transcendental models.

---

## 1. Controlled Model Configurations

Train small, matched sequence models under identical parameter counts ($1.5\text{M} \pm 2\%$, $d = 128$, 4 layers, 4 heads, $d_k = 32$):
1. **Pure Algebraic Model (Ours):**
   - Layers: ALU-GLU feed-forward networks ($\text{expansion} = 2.67d$);
   - Attention: Algebraic Attention (AA) with A-Softmax ($\kappa_8(x)$, $\Omega = 0.5$);
   - Positional Encoding: AGO Cayley rotations on $\mathfrak{so}(2)$;
   - Normalization: Parameter-free AVN;
   - Loss Functional: OACE ($\mathcal{L}_{1/8}$);
   - Optimizer: ACO with ARDS rational learning rate schedule.
2. **Standard Transformer Control:**
   - SwiGLU FFN, Exponential Softmax Attention, RoPE, RMSNorm, Cross-Entropy ($-\ln p$), AdamW + Cosine Annealing.
3. **Linear Attention / Gated DeltaNet Control:**
   - Linear attention with associative delta-rule recurrent state.
4. **Frozen Random Feature Control:**
   - Same algebraic architecture with frozen random weights (sanity check establishing that representation learning, not random projection, drives success).

### Mandatory Ablations:
- A-Softmax sharpening degree: $\kappa_2$ vs. $\kappa_4$ vs. $\kappa_8$ vs. $\kappa_{16}$;
- Attention sink parameter: $\Omega \in \{0.0, 0.2, 0.5, 1.0\}$;
- Feature charts: Shared key/query projections vs. independent linear projections;
- Normalization: Parameter-free AVN vs. learned-scale RMSNorm.

---

## 2. In-Context Reasoning Curriculum (7 Task Families)

Train and evaluate across seven distinct algorithmic and synthetic sequence reasoning task families:

1. **In-Context Linear & Affine Regression:**
   - Sequences of $(x_i, y_i)$ pairs where $y_i = \mathbf{w}^\top \mathbf{x}_i + \epsilon$. The model must predict $y_T$ given query $x_T$.
   - Evaluate Mean Squared Error (MSE) across noise levels $\sigma \in [0.0, 0.5]$.
2. **Associative Memory & Multi-Query Associative Recall (MQAR):**
   - Synthesize sequences of $N$ random key-value pairs followed by query keys.
   - Evaluate retrieval accuracy across key vocabulary size $V_K \in [64, 512]$ and distractor loads.
3. **Sequence Induction & Pattern Completion:**
   - Induction head benchmark: Sequences $[ \dots A, B, \dots A \to ? ]$. The model must identify the token $B$ following prior occurrence of $A$.
4. **Selective Copy & Token Shift:**
   - Ingest sequences with interspersed noise tokens and copy only marked target tokens to output slots.
5. **Cache-Boundary & Recency Generalization:**
   - Evaluate retrieval accuracy for targets located at positions $t - 1, t - 2, \dots, t - L$, testing whether AGO rotations preserve relative positional hierarchy.
6. **Length Extrapolation (Out-of-Distribution Context):**
   - Train on context length $L_{\text{train}} = 256$; evaluate zero-shot on $L_{\text{test}} \in \{512, 1024, 2048\}$.
7. **Negative Contexts & Adversarial Distractors:**
   - Queries with no matching key in context (testing whether attention sink $\Omega$ correctly absorbs mass to avoid false hallucinated recall).

---

## 3. Experimental Protocol & Statistical Rigor

- Execute all experiments across **five paired random seeds** (Seeds 42, 123, 456, 789, 1337) with identical training sequence streams.
- Metrics to log per run:
  - Task accuracy and cross-entropy / OACE loss;
  - Effective rank of hidden representations: $\operatorname{ER}(\mathbf{H}) = \exp(-\sum \tilde{\sigma}_i \ln \tilde{\sigma}_i)$ or algebraic equivalent $\operatorname{ER}_{\text{alg}}(\mathbf{H}) = (\sum \sigma_i)^2 / \sum \sigma_i^2$;
  - Gradient norm dynamics $\max_t \|\mathbf{g}_t\|_2$ and condition number of weight updates;
  - Convergence speed (training steps required to attain $99.0\%$ task accuracy).

---

## 4. Self-Correction & Pathology Playbook

- **Pathology: Representation Rank Collapse:**
  - *Symptom:* Effective rank $\operatorname{ER}_{\text{alg}}(\mathbf{H}) \to 1.0$; representations collapse to a 1D subspace.
  - *Diagnosis:* AVN variance scalar or attention temperature too aggressive, driving ALU into saturation.
  - *Repair:* Calibrate temperature scale $\tau = 1 / \sqrt{d_k}$, enforce Algebraic Information Preservation (AIP) anti-collapse penalty $\|\mathbf{C} - \mathbf{I}\|_F^2$.
- **Pathology: Induction Failure on Out-of-Distribution Lengths ($L > 256$):**
  - *Symptom:* Accuracy drops below $50\%$ on $L = 1024$.
  - *Diagnosis:* Cumulative rounding drift in Cayley rotation powers $(\mathbf{R}_k)^m$.
  - *Repair:* Apply algebraic normalization on Cayley matrix powers: $\mathbf{R}_k(m) \leftarrow \mathbf{R}_k(m) \cdot \operatorname{rsqrt}(\frac{1}{2}\|\mathbf{R}_k(m)\|_F^2)$.

---

## PASS Gates

- [ ] Pure Algebraic Model achieves $\ge 99.0\%$ accuracy on Associative Recall and Sequence Induction tasks on all 5 seeds.
- [ ] Convergence speed (steps to $99\%$ accuracy) is within $1.15\times$ of the Standard Transformer baseline on all 5 seeds.
- [ ] Algebraic Attention with sink $\Omega = 0.5$ correctly suppresses output on negative/no-match contexts, achieving $< 5\%$ false-positive recall vs $> 40\%$ for sink-free softmax.
- [ ] Length extrapolation: Model trained at $L=256$ retains $\ge 90.0\%$ associative recall accuracy at $L=1024$.
- [ ] Effective rank of hidden states remains non-degenerate ($\operatorname{ER}_{\text{alg}} \ge 0.60 \times d$) across all layers.
- [ ] Learned Algebraic Model strictly outperforms the frozen random feature control by $> 50\%$ absolute margin on all tasks.
- [ ] All 5 seeds (including any divergence or outlier runs) are reported in full without cherry-picking.
- [ ] All Lean 4 proofs compile cleanly via `/root/.elan/bin/lake build`.
- [ ] All inherited Phase 0–2 gates pass.
- [ ] `results/phase3/PASS.md` satisfies the shared PASS record contract.
