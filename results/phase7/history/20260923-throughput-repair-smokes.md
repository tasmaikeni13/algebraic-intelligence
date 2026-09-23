# Historical Phase 7 throughput repair smokes — 2026-09-23

These 1,000-step paired TPU runs were diagnostics only. They intentionally fail
the 100,000-step execution-budget gate, were written outside the repository,
and cannot be used as Phase 7 evidence. All ran on 16 TPU v4 devices across four
hosts using the same WikiText-103 token cache and model/data budgets.

| Diagnostic source | Algebraic tok/s | Standard tok/s | Ratio | Throughput gate |
|---|---:|---:|---:|---:|
| `af0047c` after the three-sweep OACE repair | 2,078,007 | 2,454,100 | 0.84675 | FAIL |
| causal AFA backward tile pruning | 2,209,690 | 2,441,579 | 0.90503 | PASS |
| causal pruning plus BF16/FP32-accumulating AFA dots | 2,210,538 | 2,456,788 | 0.89977 | FAIL |
| final `b3e1f5f` source, including exact Gram OACE normalization and 16,384-token tiles | 2,360,774 | 2,490,032 | 0.94809 | PASS |

The final short run also recorded a perplexity ratio of 1.00097, zero non-finite
updates, zero loss spikes, and a peak clipped gradient norm of 1.00000024. The
component benchmark at the exact Phase 7 per-chip attention shape measured AFA
forward/backward at 0.644 ms and maintained TPU FlashAttention at 0.645 ms. It
therefore localized the remaining full-step difference to the vocabulary head;
the larger bounded OACE tile then supplied stable throughput headroom.

Only a clean, source-matched 100,000-step run followed by the Phase 7 aggregate
verifier may replace these diagnostics with current evidence.
