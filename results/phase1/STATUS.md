# Phase 1: CPU verified; TPU validation deferred

The implemented Phase 1 CPU and formal gates pass: **33 unit tests**, the complete
Lean build, the source/graph purity audit, the float64 identities, million-sample
Monte Carlo experiments at seven scales, and **10,000 independent trials at each
of depths 8, 16, 24, and 32** (width 128).

**Phase 1 remains incomplete.** The user deferred TPU use until 2026-09-16.
The required 16-chip TPU float32/BF16 execution and performance comparisons have
not run. No `PASS.md` is issued and no TPU job is scheduled.

- [Metrics and source fingerprints](metrics.json)
- [Lean build](lean-build.log) and [unit tests](pytest.log)
- [Complete CPU trial tensors](deep-trials.npz)
- [Statistical summary](SUMMARY.md) and [high-resolution figure](verification.png)
- [Failures, repairs, and amendments](ITERATIONS.md)
- [Reproduction and TPU launch commands](REPRODUCE.md)

Verified implementation snapshot: `3e590319285bf3bf760b02815ca3ef8201c7c753`. The original run began before
that commit and records its actual dirty worktree and base revision; every source
hash was checked against this committed snapshot after the run. The raw record
is preserved unchanged in `iterations/full-cpu-width128/` and copied here for
convenience. CPU smoke timings are explicitly excluded from TPU evidence.
