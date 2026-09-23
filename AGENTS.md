# Repository Guide for Coding Agents

This repository implements and tests a JAX research stack that replaces selected Transformer transcendentals with algebraic primitives, alongside a compute-matched standard Transformer. Mathematical claims, production code, hardware kernels, phase protocols, and result artifacts must agree; prose and historical result files are never evidence by themselves.

## Where to Work

- `src/`: model, optimizer, dataset, mesh, and mathematical primitives.
- `src/kernels/`: Pallas/XLA TPU kernels, Triton GPU kernels, and fused vocabulary losses.
- `tests/`: numerical references, regression tests, evidence-contract tests, and static audits.
- `scripts/`: local verification, dataset preparation, TPU launchers, training, and evidence aggregation.
- `phases/`: authoritative execution budgets and PASS gates for each research phase.
- `formal/`: Lean 4 certificates. Do not edit generated `.lake` content.
- `results/`: generated evidence. A file named `PASS.md` is valid only when its machine-readable evidence matches current source hashes and every phase gate.

## Environment and Commands

Use the checked-in virtual environment when present:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python scripts/smoke_phase8.py
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python scripts/smoke_phase9.py
cd formal && lake build
```

Run the smallest relevant test module while iterating, then the full suite before committing. Force the CPU backend for local tests: this workspace may share a TPU runtime, and ordinary tests must not claim or disturb it. TPU launch commands and long training runs are documented in the corresponding `phases/phaseN.md` file.

## Scientific and Implementation Invariants

- The algebraic production stack must pass `scripts.audit_primitives.source_audit`: no `exp`, `log`, trigonometric, hyperbolic, sigmoid, or non-integer-power calls. Standard-baseline and evaluation code may use their required transcendental functions.
- `AlgebraicTransformerLM` must use Algebraic FlashAttention and fused Linear-OACE in training. `StandardTransformerLM` must use maintained TPU FlashAttention when available, the tested tiled fallback elsewhere, and fused linear cross-entropy.
- Attention kernels must preserve causal masking for unequal query/key tile sizes, arbitrary sequence lengths through valid-token padding, FP32 accumulation for reduced-precision inputs, and correct gradients for Q, K, and V.
- Do not describe the diagonal-feature recurrence in `linear_afa.py` as exact octic AFA. It is a Taylor-truncated approximation and is outside the Phase 7–9 production path.
- Keep both architectures matched in parameter count, token order, batch size, context length, seed set, and processed-token budget. Full-batch budgets round upward; never record requested tokens as processed tokens.
- Smoke tests validate wiring only. They cannot create or refresh PASS evidence.
- Phase 7 requires 100,000 steps per architecture on 16 TPU v4 chips. Phase 8 requires every preregistered candidate in `phases/phase8_candidates.json` on seeds 42, 43, and 44 with at least 600M tokens per run. Phase 9 must reject stale or invalid Phase 8 winners.
- Preserve failed or superseded measurements as clearly labeled historical records. Never manufacture, copy, or hand-edit measured values into current evidence.

## Evidence and Hardware Runs

Commit the exact source snapshot before a distributed launcher creates its Git bundle. Launchers must refuse dirty worktrees, detached heads, busy accelerators, missing datasets, and incomplete result collections. Copy datasets to every TPU worker, including worker 0. Hardware evidence must record source hashes, topology, requested and actual budgets, numerical safety counters, and complete candidate/seed coverage.

When source files in an evidence dependency closure change, expect the old artifact to become stale. Regenerate the relevant hardware evidence or report the phase as requiring a rerun; do not weaken hash checks or inherited gates to preserve an earlier PASS.

## Code and Review Conventions

Use plain Python with type hints where they clarify serialized records or public interfaces. Match nearby JAX patterns, keep static configuration outside traced array arguments, and make reductions explicit about accumulation dtype. Add regression tests for numerical or indexing bugs; avoid tests that only mirror implementation details.

Before finishing a change, inspect `git diff`, run the relevant smoke and test commands, verify JSON can be parsed with `allow_nan=False` semantics, and update phase documentation when a command, budget, artifact schema, or architecture path changes. Do not commit datasets, virtual environments, compiler caches, or Phase 9 checkpoint payloads.
