# Historical incomplete Phase 7 run — OACE throughput failure — 2026-09-23

This run used commit `1d7e7741137686834cdfc85725522464afcde764` on 16 TPU v4
devices across four hosts. It is an incomplete failed diagnostic and is not
Phase 7 evidence. No `metrics.json` or `PASS.md` was produced.

The standard arm completed all 100,000 steps in 1,328.8 seconds. Its final
logged throughput was 2,471,287 tokens/second, its final loss was 2.9112, and
its held-out perplexity was 25.23. The algebraic arm was stopped after step
3,500 because its timing had stabilized near 2,034,000 tokens/second, or about
82.3% of the standard arm. Continuing could not satisfy the preregistered 90%
throughput gate. The algebraic loss was finite and fell from 27.9953 at step
500 to 15.1393 at step 3,500; the clipped gradient norm was 1.000 throughout
the logged interval.

Inspection found that fused Linear-OACE projected over the full vocabulary
four times per optimizer update: twice in forward, once to reconstruct the AVN
radial reduction, and once to emit gradients. The radial reduction is an exact
linear combination of three compact sums already available during the second
forward pass. The correction accumulates those sums in forward, caches the
per-token radial scalar, and reduces backward to one vocabulary pass. Numerical
loss, gradient, arbitrary-vocabulary-tail, and zero-transcendental regression
tests cover the revised three-sweep implementation. Phase 7 must be rerun from
the corrected committed source; none of the measurements above can be promoted
to current evidence.
