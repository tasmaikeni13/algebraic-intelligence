# Phase 7 — Matched Multi-Seed Scaling Study (125M Parameters on 1.0B FineWeb-Edu Tokens)

Start only after Phase 6 PASS. Read all prior artifacts and `phases/AUTONOMY_PROTOCOL.md`. Execute the failure-repair loop until PASS.

This phase executes the definitive empirical head-to-head pretraining comparison at the **125M parameter scale** across **1.0 Billion tokens of FineWeb-Edu** on the dedicated **1x AMD Instinct MI300X GPU (192 GB HBM3)**. It tests whether pure algebra matches or exceeds transcendental architectures under identical data and optimization budgets.

---

## 1. Frozen Experimental Generation

Preregister and freeze the experimental configuration before launching training runs:

- **Model Scale:** **125M parameters** ($d_{\text{model}} = 768$, $\text{heads} = 12$, $\text{layers} = 12$, $d_k = 64, d_v = 64$, FFN dimension $2048$, vocabulary size $50,257$ using GPT-2 / FineWeb-Edu standard tokenizer);
- **Dataset & Token Budget:** Exactly **1.0 Billion training tokens** per model drawn from the **FineWeb-Edu** corpus (`HuggingFaceFW/fineweb-edu`);
- **Hardware Target:** Dedicated 1x AMD Instinct MI300X GPU (192 GB HBM3, CDNA3 `gfx942`);
- **Architectures Compared:**
  1. **Pure Algebraic Transformer (`AlgebraicTransformerLM`):** ALU-GLU, A-Softmax ($\kappa_8$, $\Omega=0.5$), AGO Cayley rotations, AVN, OACE ($\mathcal{L}_{1/8}$), ACO factorized curvature + ARDS schedule;
  2. **Standard Causal Transformer Baseline:** SwiGLU, Exponential Softmax Attention, RoPE, RMSNorm, Cross-Entropy ($-\ln p$), AdamW + Cosine Annealing;
  3. **Competitive SSM-Attention Hybrid:** Alternating Mamba-2 style selective scan and causal attention blocks, SwiGLU, RMSNorm, AdamW;
- **Paired Seeds:** Run all three architectures across **three identical random seeds** (Seed 42, Seed 1337, Seed 2026), yielding $3 \times 3 = 9$ complete pretraining runs;
- **Training Protocol:** Identical FineWeb-Edu shard ordering, global batch size $\approx 1.05 \times 10^6$ tokens (512 sequences of context length 2048), BF16 precision with FP32 master weights and optimizer state accumulation, checkpoint cadence every 50M tokens;
- **Budget Parity:** Parameters within $\pm 1\%$, training tokens identical, optimizer step counts identical.

---

## 2. Evaluation Suite & Statistical Protocol

Evaluate all 9 completed runs across:

1. **Language Modeling Perplexity:**
   - Validation perplexity and loss on held-out FineWeb-Edu validation split;
   - Report mean $\pm$ standard error of the mean (SEM) and 95% confidence intervals across seeds.
2. **Downstream Zero-Shot Reasoning:**
   - Standard lm-evaluation-harness benchmarks: ARC-Easy, HellaSwag, PIQA, and LAMBADA;
   - Report mean accuracy $\pm$ SEM across seeds.
3. **Training Stability Dynamics:**
   - Maximum gradient norm $\max_t \|\mathbf{g}_t\|_2$ over 1B tokens;
   - Count of loss spikes ($\Delta \mathcal{L} > 1.5$) and non-finite (NaN / Inf) iterations.
4. **Systems & Efficiency Telemetry on MI300X:**
   - Sustained training throughput (tokens/second);
   - Peak VRAM allocation;
   - Optimizer memory state bytes: Confirm that ACO reduces second-moment memory from $\approx 500\text{ MB}$ to $< 1\text{ MB}$, saving $\ge 45\%$ total optimizer memory in HBM.

---

## 3. Failure Repair & Regression Protocol

If any seed diverges, experiences loss spikes, or fails to meet the parity threshold:
1. Freeze the trace, checkpoint, and numerical state;
2. Classify the root cause (e.g. learning rate warmup too short, rational decay parameter $\alpha$ miscalibrated, gradient accumulation precision loss);
3. Derive an algebraic correction, formalize any altered deterministic lemma in Lean 4, verify that `lake build` passes, and run Phase 1–6 regression tests;
4. Rerun all 3 seeds under symmetric budgets. Selective exclusion of unfavorable seeds is strictly forbidden.

---

## PASS Gates

- [ ] All 9 pretraining runs (3 architectures $\times$ 3 seeds) complete the full 1.0B token budget with zero unhandled NaNs or divergent loss spikes.
- [ ] Algebraic Transformer validation perplexity on FineWeb-Edu achieves parity with the Standard Transformer baseline within $\le 1.08\times$ (mean over 3 paired seeds).
- [ ] Perplexity variance across seeds is low and stable: $\operatorname{SEM} \le 0.15$.
- [ ] Downstream zero-shot reasoning benchmarks (ARC-Easy, HellaSwag, PIQA, LAMBADA) are within $2.0\%$ absolute margin of the Standard Transformer baseline.
- [ ] Hardware measurements on 1x MI300X confirm $\ge 45\%$ lower optimizer memory footprint for ACO compared to AdamW.
- [ ] Strict Zero-Transcendental audit confirms 0 transcendental function calls across all 125M Algebraic checkpoints and training traces.
- [ ] All Lean 4 formal proofs compile cleanly via `/root/.elan/bin/lake build`.
- [ ] All inherited Phase 0–6 gates pass.
- [ ] `results/phase7/PASS.md` satisfies the shared PASS record contract with complete reproduction logs.
