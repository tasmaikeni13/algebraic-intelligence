# Phase 4 — Long-Context Extrapolation, Compositional Reasoning, & Sub-Byte Quantization Gate

Start only after Phase 3 PASS. Read all prior artifacts and `phases/AUTONOMY_PROTOCOL.md`. Execute the mandatory failure-repair loop until PASS.

This phase stresses the Algebraic Stack under extreme conditions: **long-context horizon scaling ($16\times$ training length), multi-hop compositional pointer chasing, non-stationary concept drift, and sub-byte (FP4 / INT4) hardware quantization**.

---

## 1. Required Stress Suites

### 1.1 Multi-Hop Compositional Pointer Chasing
Evaluate the model's ability to traverse chained relational pointers $[ k_0 \mapsto v_0, k_1(=v_0) \mapsto v_1, \dots, k_H \mapsto v_H ]$:
- **Hop Horizons:** $H \in \{1, 2, 4, 8, 16\}$ sequential pointer hops;
- **Distractor Density:** Interleave 10, 50, and 100 distractor key-value pairs between legitimate chain links;
- **Metrics:** Trace error propagation $\| \mathbf{h}_H - \mathbf{v}_H^* \|_2$, decoded symbol accuracy, and attention entropy per hop.
- **Hypothesis:** Pure algebraic composition avoids exponential dynamic range explosion, maintaining stable error propagation across multi-hop chains.

### 1.2 Extreme Sequence Length Extrapolation ($16\times$)
Evaluate models trained at context length $L_{\text{train}} = 256$ and $L_{\text{train}} = 2048$ on sequence lengths extending up to $16\times$ horizon:
- $L_{\text{test}} \in \{512, 1024, 2048, 4096, 8192, 16384\}$ tokens;
- **AGO Rotational Stability:** Measure cumulative angular drift in Cayley rotations $\mathbf{R}_k(m) = (\mathbf{R}_k)^m$. Verify that no frequency aliasing or catastrophic norm degradation occurs at step $16,384$.
- **Associative Memory Contractivity:** Verify Theorem 8.2: Under normalized keys $\|\mathbf{k}_t\| = 1$, the global associative memory state $\|\mathbf{S}_t\|_F$ remains strictly bounded by $\gamma_{\max} V / f_{\min} < \infty$, preventing recurrent state blowup.

### 1.3 Non-Stationary Dynamics & Concept Drift
- **Abrupt Key-Value Overwrites:** Update key $K$ with new value $V_{\text{new}}$ midway through the sequence. Measure the model's ability to recall the latest update while discarding stale history.
- **Corrupted Logits & Outlier Injections:** Inject extreme magnitude outliers ($100\sigma$) into isolated token embeddings. Verify that AVN pre-bounding bounds coordinates $|\hat{x}_i| \leq \sqrt{d}$, preventing layer collapse.

### 1.4 Native Sub-Byte Quantization Stress (FP4 / INT4 / FP8)
Simulate hardware-native sub-byte quantization on the model's forward pass:
- **Formats:** IEEE FP8 (E4M3 and E5M2), INT4 (affine min-max), and microscopic FP4 (E2M1);
- **Quantization Targets:** Weights, activation tensors, and attention logit matrices;
- **Comparison:** Pure Algebraic Model vs. Standard Transcendental Transformer;
- **Hypothesis to Validate:** Because the A-Softmax kernel $\rho$ is globally 2-Lipschitz, logit quantization noise propagates additively, not exponentially ($\operatorname{Var}(\rho(X)) \leq 4 \operatorname{Var}(X)$). The Algebraic Stack runs stably in FP4/INT4 without requiring dynamic per-group scales or QAT outlier suppression.

---

## 2. Failure Diagnostics & Playbook

- **Symptom: Error blows up exponentially with hop count $H \ge 8$:**
  - *Root Cause:* Un-normalized residual connections accumulating variance across multiple forward passes.
  - *Correction:* Insert AVN normalization on intermediate recurrent pointer reads.
- **Symptom: Extreme degradation ($> 5.0$ loss spike) under FP4 quantization:**
  - *Root Cause:* Intermediate accumulator precision too low in A-Softmax denominator.
  - *Correction:* Ensure block-level denominator summation is accumulated in FP32 scratch registers while storing inputs and outputs in FP4/INT4.

---

## 3. PASS Gates

- [ ] Multi-hop pointer chasing achieves $\ge 90.0\%$ decoded accuracy at $H = 4$ hops and $\ge 75.0\%$ at $H = 8$ hops under distractor load.
- [ ] Model trained at $L=256$ maintains stable perplexity and zero non-finite outputs when extrapolated to $L=4096$ ($16\times$ horizon).
- [ ] Contractive memory norm $\|\mathbf{S}_t\|_F$ is proven and empirically confirmed strictly bounded across 16,384 consecutive update steps.
- [ ] Post-training FP4 quantization degradation on validation loss is $\le 0.40$ units for the Algebraic Stack (vs $\ge 2.50$ units for the Standard Softmax Transformer).
- [ ] INT4 attention scores maintain $> 95\%$ of unquantized associative recall accuracy without per-group dynamic scale vectors.
- [ ] Zero loss spikes, NaNs, or gradient explosions under abrupt key-value overwrites and extreme outlier injections.
- [ ] All Lean 4 formal proofs compile cleanly via `/root/.elan/bin/lake build`.
- [ ] All inherited Phase 0–3 gates pass.
- [ ] `results/phase4/PASS.md` satisfies the shared PASS record contract.
