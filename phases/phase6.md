# Phase 6 — Language-Model Viability & Head-to-Head Comparative Publication Gate

Start only after Phase 5 PASS. Read all prior evidence and `phases/AUTONOMY_PROTOCOL.md`.

This phase establishes natural language modeling viability and rigorous comparative benchmarks for publication across **two matched architectures: The Pure Algebraic Stack vs. The Modern Causal Transformer at both 125M and 350M parameter scales**. Execute the mandatory failure-repair loop until PASS.

---

## 1. Two Publication Candidate Architectures

For peer-reviewed publication and definitive architectural comparison, implement and evaluate two matched candidates:

1. **Candidate 1: The Pure Algebraic Stack (`AlgebraicTransformerLM`)**:
   - Gating: ALU-GLU feed-forward blocks ($\text{expansion} = 2.67d$) with exact Horner cubic backward pass;
   - Attention: A-Softmax with octic kernel $\kappa_8(x)$, attention sink $\Omega = 0.5$, and hardware-accelerated AFA additive accumulation;
   - Positional Encoding: AGO Cayley rotations on $\mathfrak{so}(2)$, providing exact shift equivariance without trigonometry;
   - Normalization: Parameter-free AVN, eliminating learnable $\boldsymbol{\gamma}$ from HBM;
   - Loss Functional: Octo-Algebraic Cross-Entropy (OACE / $\mathcal{L}_{1/8}$);
   - Optimization: Factorized Algebraic Curvature Optimizer (ACO) with ARDS rational decay schedule;
   - Zero transcendental operations throughout the entire training and inference cycle.

2. **Candidate 2: Modern Causal Transformer (Standard Transcendental Baseline)**:
   - Modern LLaMA/Mistral-style decoder-only architecture;
   - Rotary Position Embeddings (RoPE), Pre-RMSNorm, Causal Multi-Head Self-Attention, and SwiGLU MLP ($d_{\text{ffn}} = \frac{8}{3}d_{\text{model}}$);
   - Standard Cross-Entropy loss ($-\ln p$), AdamW optimizer with Cosine Annealing;
   - Standard $O(L)$ growing KV cache at inference and $O(L^2)$ training complexity.

---

## 2. Dual Model Capacity Targets: 125M and 350M

Preregister and implement both parameter scales to validate small-scale viability and medium-scale scaling readiness:

- **125M Parameter Scale:**
  - $d_{\text{model}} = 768$, $\text{heads} = 12$, $\text{layers} = 12$, $d_k = 64, d_v = 64$;
  - Vocabulary: $50,257$ (GPT-2 / FineWeb-Edu tiktoken standard);
  - Context length: $2048$ tokens.
- **350M Parameter Scale:**
  - $d_{\text{model}} = 1024$, $\text{heads} = 16$, $\text{layers} = 24$, $d_k = 64, d_v = 64$;
  - Vocabulary: $50,257$;
  - Context length: $2048$ tokens.

Parameters across both candidates must be calibrated within $\pm 1\%$ at each scale.

---

## 3. Diagnostic & Natural Language Benchmark Suite

Evaluate both candidates across both scales on:

1. **FineWeb-Edu Token Distribution:** Validation loss, convergence trajectory, and perplexity on held-out **FineWeb-Edu** tokens.
2. **Multi-Query Associative Recall (MQAR):** Key-value retrieval across varied distractor loads and sequence lengths.
3. **Induction & Selective Copy:** Long-distance prefix pattern completion and selective token extraction.
4. **Cache-Boundary & Recency Generalization:** Retrieval precision across local window eviction boundaries.
5. **Multi-Hop Pointer Chains:** Chained associative pointer chasing through $\{2, 4, 8\}$ hops.
6. **Passkey Retrieval / Needle-In-A-Haystack (NIAH):** Accurate passkey retrieval at sequence lengths up to 4096 tokens.
7. **Systems Profiling (1x AMD Instinct MI300X):**
   - Prefill throughput (tokens/second) across lengths $\{512, 1024, 2048, 4096\}$;
   - Per-token decode latency (ms/token);
   - Peak VRAM allocation during training and inference;
   - Second-moment optimizer memory footprint: ACO factorized $\mathcal{O}(d_{\text{out}} + d_{\text{in}})$ vs AdamW $\mathcal{O}(d_{\text{out}} \cdot d_{\text{in}})$.

---

## 4. Failure Repair & Calibration Playbook

- If the Algebraic Transformer exhibits higher initial loss than the baseline, do not abandon the pure algebraic formulation. Verify that the rational learning rate warmup $T_{\text{warm}}$ is calibrated and check that the attention sink $\Omega$ is not set too large ($\Omega \in [0.1, 0.5]$).
- If gradient norms fluctuate in early training, calibrate the ARDS decay parameter $\alpha$ and verify that the AVN variance scalar regularizer $\epsilon = 10^{-6}$ prevents division by zero in zero-variance token embeddings.
- Every architectural adjustment must preserve the Zero-Transcendental Axiom and compile cleanly in Lean 4.

---

## PASS Gates

- [ ] Both architectures (Algebraic Stack and Standard Causal Transformer) are fully implemented, calibrated at both 125M and 350M scales ($\pm 1\%$ params), and pass all gradient checks.
- [ ] Accelerated kernels compile and pass numerical validation against fp64 CPU references with maximum relative error $< 1.0 \times 10^{-5}$ in float32.
- [ ] Algebraic Transformer achieves validation perplexity on FineWeb-Edu token distributions within $8\%$ parity ($\le 1.08\times$) of the Standard Transformer baseline at 125M scale.
- [ ] Algebraic Transformer demonstrates statistically significant advantage over standard Transformer on sub-byte FP4 quantization stability.
- [ ] Multi-Query Associative Recall (MQAR) and multi-hop pointer chasing benchmarks confirm parity with the Standard Transformer baseline.
- [ ] Systems profiling on the MI300X confirms that ACO achieves $\ge 45\%$ lower total optimizer memory consumption compared to AdamW at 125M and 350M scales.
- [ ] AST code audit confirms exactly 0 transcendental operations in the Algebraic Transformer training and inference loops.
- [ ] All Lean 4 formal proofs compile cleanly via `/root/.elan/bin/lake build`.
- [ ] All inherited Phase 0–5 gates pass.
- [ ] `results/phase6/PASS.md` satisfies the shared PASS record contract.
