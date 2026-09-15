# Phase 1 dependency audit

Only Phase 1 is implemented and executed.

| Consumer | Repository state at this audit | Contract carried forward |
| --- | --- | --- |
| Phase 2 `src/attention.py` | Not implemented | AVN normalizes the last axis without centering; eps=1e-5 remains unchanged. The coordinate norm bound survives the gate amendment. |
| Phase 6 `src/kernels/pallas_afa.py` | Not implemented | BF16 inputs use FP32 accumulation/cache; output and cotangent dtypes match input. ALU caches only u; AVN caches normalized coordinates and tau. |
| Phases 7–9 `src/model.py` | Not implemented | Residual depth attenuation is algebraic. Phase 1 tests a normalized two-matrix residual network; its empirical bounds do not certify a future ALU-GLU transformer. |
| All future variance monitors | Specifications only | Record centered variance and second moment separately; never infer variance=1 from an RMS normalization. |

No downstream signature changed: `alu(x)` and `avn(x, eps=1e-5)` retain the
specified interface. The negative-tail repair changes only an equivalent forward
expression. No later-phase placeholder modules were created. The future residual
attenuation mechanism is already described in `phases/phase7.md` §7 and
`phases/phase8.md` §7. Future assemblies must remeasure their actual architecture.
