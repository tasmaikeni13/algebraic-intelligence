# Superseded Phase 8 Hyperparameter Sweep (2026-09-25)

## Status: SUPERSEDED / INVALIDATED

This run (recorded in commit `a491f1d`) was superseded due to critical methodological flaws that biased the hyperparameter sweep against the Standard Transformer baseline:

1. **Warmup Loss Spike Filter Bug**:
   - `scripts/run_hparam_sweep.py` checked `step >= 10 and (loss_val - prev_loss) > 1.5` with the comment `excluding initial warmup`.
   - The actual warmup duration was 28 steps (reference/lower_lr) and 57 steps (high_lr).
   - Natural stochastic batch variance during warmup caused single-step fluctuations > 1.5, triggering `loss_spike_count = 1` for `base_high_lr` (seed 43) and `base_reference` (seeds 42, 43, 44).
   - This erroneously disqualified the best-performing baseline candidate (`base_high_lr`, mean validation perplexity 55.72) and forced the selection of `base_lower_lr` (mean validation perplexity 126.76) as the baseline "winner" with learning rate 0.0003.

2. **Weight Decay on RMSNorm Gammas**:
   - Decoupled weight decay was applied without a parameter mask, causing 1D RMSNorm scale parameters (`norm1_gamma`, `norm2_gamma`, etc.) in the baseline to decay steadily (weight decay 0.10 in `base_high_lr`), suppressing residual stream activations.
   - The Algebraic Transformer uses parameter-free AVN and was unaffected.

3. **PRNG Key Splitting Modulo Recycling**:
   - `init_params` in both architectures recycled 20 or 25 keys with modulo arithmetic, causing 52 duplicated weight matrices across layers in the baseline and 30 in the algebraic model.

All three defects have been corrected in source code. This directory preserves the historical measurements per repository governance invariants.
