# Phase 8: Frontier Pretraining: 125M Parameters on 1.0B FineWeb-Edu Tokens

Start only after Phase 7 PASS. Read `theory.md`, Phase 7 evidence in `results/phase7/` and `phases/README.md` completely before executing. Execute the shared failure-repair loop until all gates pass.

---

## 1. Objective, Scientific Hypothesis & Competing Models

Execute the definitive empirical head-to-head pretraining comparison at the **125M parameter scale** across **1.0 Billion tokens of FineWeb-Edu** on the dedicated **1x AMD Instinct MI300X GPU (192 GB HBM3)**:
$$\textbf{"Can pure algebra match or exceed the standard causal Transformer at the 125M / 1.0B token frontier?"}$$

### Competing Hypotheses:
- **$H_1$ (Algebraic Hypothesis):** The 125M Pure Algebraic Transformer (`AlgebraicTransformerLM`) achieves validation perplexity parity ($\le 1.08\times$) and downstream zero-shot reasoning parity (within $2.0\%$ absolute margin on ARC-Easy, HellaSwag, PIQA, and LAMBADA) relative to the Standard Causal Transformer baseline across 3 paired seeds, while maintaining bounded gradient dynamics, zero loss spikes, and $\ge 45\%$ lower optimizer memory in local HBM3.
- **$H_0$ (Transcendental Baseline Hypothesis):** On a large web corpus (1.0B tokens), continuous transcendental functions (Swish activation, exponential Softmax, RoPE trigonometric embeddings, cross-entropy $-\ln p$, and AdamW + Cosine schedule) provide essential inductive advantages that pure algebraic approximations cannot replicate, resulting in diverging validation loss, representation collapse, or severe downstream benchmark degradation.

---

## 2. Matched Multi-Seed Experimental Configuration & Budget Parity

Preregister and freeze the experimental configuration before launching training runs:

```mermaid
graph LR
    subgraph "Hardware Substrate"
        GPU["1x AMD Instinct MI300X (192 GB HBM3, CDNA3 gfx942)"]
    end

    subgraph "Candidate 1: Pure Algebraic Transformer (125M)"
        M1["AlgebraicTransformerLM 125M<br/>(ALU-GLU, AVN, A-Softmax, AGO, OACE, ACO)"]
        S1a["Seed 42"] --> M1
        S1b["Seed 1337"] --> M1
        S1c["Seed 2026"] --> M1
    end

    subgraph "Candidate 2: Standard Causal Transformer (125M)"
        M2["StandardTransformerLM 125M<br/>(SwiGLU, RMSNorm, Exp Softmax, RoPE, CE, AdamW)"]
        S2a["Seed 42"] --> M2
        S2b["Seed 1337"] --> M2
        S2c["Seed 2026"] --> M2
    end

    GPU --- M1
    GPU --- M2
```

### 2.1 Model Specifications (125M Scale)
- **Parameter Count:** $\approx 125\text{M}$ parameters (matched within $\pm 1\%$ across architectures).
- **Hidden Dimension ($d_{\text{model}}$):** $768$.
- **Number of Layers ($L$):** $12$.
- **Attention Heads ($H$):** $12$ ($d_k = d_v = 64$ per head).
- **FFN Intermediate Dimension ($d_{\text{ff}}$):** $2048$ ($8/3 \times d_{\text{model}} \approx 2048$).
- **Vocabulary Size ($V$):** $50,257$ (GPT-2 standard BPE tokenizer).
- **Context Length ($T$):** $2048$ tokens.
- **Dataset:** Exactly **1.0 Billion training tokens** drawn from the **FineWeb-Edu** corpus (`HuggingFaceFW/fineweb-edu`).
- **Global Batch Size:** $\approx 1.05 \times 10^6$ tokens ($512$ sequences $\times 2048$ context length).
- **Precision:** BF16 mixed-precision with FP32 master weights and optimizer state accumulation.
- **Checkpoint Cadence:** Checkpoints saved every $50\text{M}$ tokens.
- **Paired Seeds:** 3 identical random seeds (Seed 42, Seed 1337, Seed 2026), yielding $2 \times 3 = 6$ complete pretraining runs.

### 2.2 Budget Parity Enforcement
- Parameters within $\pm 1\%$.
- Exact same token streams and shard ordering across corresponding seeds.
- Identical optimizer step counts.
- Dedicated single-socket execution on 1x AMD Instinct MI300X with zero distributed pipeline stalls.

---

## 3. Evaluation Suite & Statistical Protocol

Evaluate all 6 completed runs across the following four evaluation axes:

### 3.1 Language Modeling Perplexity
- Validation perplexity and loss on held-out FineWeb-Edu validation split.
- Report mean $\pm$ standard error of the mean (SEM) and 95% confidence intervals across the 3 paired seeds.

### 3.2 Downstream Zero-Shot Reasoning Benchmarks
Evaluate checkpoints using the standard `lm-evaluation-harness` suite:
- **ARC-Easy:** Elementary science reasoning.
- **HellaSwag:** Grounded commonsense reasoning.
- **PIQA:** Physical interaction question answering.
- **LAMBADA:** Broad narrative context word prediction.
- Report mean accuracy $\pm$ SEM across seeds. Parity bound: within $2.0\%$ absolute margin of the baseline.

### 3.3 Training Stability Dynamics
- Maximum gradient norm: $\max_{t} \|\mathbf{g}_t\|_2$ over the entire 1.0B token trajectory.
- Count of sudden loss spikes ($\Delta \mathcal{L} > 1.5$) and non-finite iterations ($0$ permitted).
- Hidden activation variance tracking across all 12 layers: verify that parameter-free AVN maintains $\operatorname{Var}(\mathbf{h}_\ell) \in [0.8, 1.3]$ without gain drift.

### 3.4 Systems & Efficiency Telemetry on MI300X
- Sustained training throughput (tokens/second) on MI300X.
- Peak VRAM allocation during training.
- Optimizer memory state bytes: Confirm that ACO reduces second-moment memory from $\approx 500\text{ MB}$ to $< 1\text{ MB}$, saving $\ge 45\%$ total optimizer memory in HBM.

---

## 4. Lean 4 Formal Verification Gate

Re-verify formal proof integrity under parameter scaling:
- Compile all modules in `formal/AlgebraicTheory/` via `/root/.elan/bin/lake build`.
- Verify that dimensional scaling ($d_{\text{model}} = 768$, $L = 12$) preserves all algebraic Lipschitz bounds, variance bounds, and loss propriety theorems.

---

## 5. Autonomous Failure Ledger & Self-Correction Playbook

- **Symptom: A single seed diverges or experiences numerical instability:**
  - *Action:* Freeze raw trace, checkpoint, and numerical state. Classify root cause (e.g. rational learning rate warmup too short, rational decay parameter $\alpha$ miscalibrated, gradient accumulation precision loss).
  - *Correction:* Derive an algebraic correction, formalize any altered deterministic lemma in Lean 4, verify that `lake build` passes, and run Phase 1–7 regression tests.
  - *Rule:* Rerun all 3 seeds under symmetric budgets. Selective exclusion of unfavorable seeds is strictly forbidden.
- **Symptom: Attention entropy collapses at sequence length 2048:**
  - *Root Cause:* Score magnitude scaling $\tau$ fails to control logit variance at long sequence lengths.
  - *Correction:* Confirm query-key scaling $\tau = \operatorname{rsqrt}(d_k)$ and calibrate attention sink parameter $\Omega \in [0.25, 1.0]$.
- **Symptom: Downstream benchmark performance gap $> 2.0\%$:**
  - *Root Cause:* Preconditioning lag in ACO factorized row/column accumulators during early training.
  - *Correction:* Implement rational debiasing factors $\hat{\mathbf{m}}_t / (1 - \beta_1^t)$ with algebraic polynomial powers.

---

## 6. PASS Gates

- [ ] All 6 pretraining runs (2 architectures $\times$ 3 seeds) complete the full 1.0B token budget with zero unhandled NaNs or divergent loss spikes.
- [ ] Algebraic Transformer validation perplexity on FineWeb-Edu achieves parity with the Standard Transformer baseline within $\le 1.08\times$ (mean over 3 paired seeds).
- [ ] Perplexity variance across seeds is low and stable: $\operatorname{SEM} \le 0.15$.
- [ ] Downstream zero-shot reasoning benchmarks (ARC-Easy, HellaSwag, PIQA, LAMBADA) are within $2.0\%$ absolute margin of the Standard Transformer baseline.
- [ ] Hardware measurements on 1x MI300X confirm $\ge 45\%$ lower optimizer memory footprint for ACO compared to AdamW.
- [ ] Strict Zero-Transcendental audit confirms 0 transcendental function calls across all 125M Algebraic checkpoints and training traces.
- [ ] All Lean 4 formal proofs compile cleanly via `/root/.elan/bin/lake build`.
- [ ] All inherited Phase 1–7 gates pass without regression.
- [ ] `results/phase8/PASS.md` satisfies the shared PASS record contract with complete reproduction logs.
