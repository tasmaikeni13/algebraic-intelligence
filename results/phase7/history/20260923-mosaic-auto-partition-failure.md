# Historical failed Phase 7 preflight — 2026-09-23

This run used commit `6af306ae88aa38590e14abcd0647f673d708ac5e` and stopped before the first optimizer update. It produced no training measurements and is not Phase 7 evidence.

All four TPU hosts passed the CPU test suite. The first standard-model compilation then failed because JAX's maintained Mosaic/Pallas FlashAttention custom call was nested directly in a mesh-wide SPMD `jit`. JAX reported:

```text
NotImplementedError: Mosaic kernels cannot be automatically partitioned. Please wrap the call in a shard_map.
```

The correction places the full training step inside an explicit data-parallel `shard_map`, averages loss and gradients across the `data`, `fsdp`, and `model` mesh axes before the optimizer update, and applies the same execution path to Phases 7, 8, and 9. A regression test executes the standard training step through that boundary. The launcher also creates the remote measurement directory before execution and preserves the original training exception if result recovery is incomplete.
