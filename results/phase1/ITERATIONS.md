# Phase 1 failure and repair ledger

All paths below are relative to `results/phase1/`. Seed is 42 unless stated.
Historical artifacts remain unchanged. A failure is never counted as a PASS.

| Iteration | Direct evidence | Diagnosis and action |
| --- | --- | --- |
| Original AVN variance gate | `iterations/variance-v1.json` | Default eps=1e-5 at sigma=0.1 produces variance 0.9990019521851182. The uncentered, regularized formula contradicts universal unit variance. Gate v2 tests exact moment and variance identities, and retains the original near-unit interval at eps=0. Public epsilon is unchanged. |
| Direct ALU negative tail | `iterations/pytest-initial.log` | 28 tests passed, two boundary tests failed. Direct 1+u cancellation amplified BF16/FP32 rounding to a large negative artifact at −1e15. Rationalized forward tail; retained the one-u Horner VJP. |
| Tail repair | `iterations/pytest-tail-repair.log` | All 30 primitive tests passed after the repair. No thresholds changed. |
| Initial Lean build | `iterations/lean-initial.log` | Existing project compiled successfully. Actual toolchain was 4.34.0-rc2, contrary to older prose. Corrected documentation. |
| Stronger Lean certificates | `iterations/lean-additions.log`, `iterations/lean-coupling.log` | Added both-inflection iff, tail identity, derivative bound, positive scaling, actual gate coupling, and exact regularized moment/damping identities. Clean build. |
| Non-residual mechanism ablation | `iterations/deep-unattenuated.json`, `.npz` | 200 trials per depth, width 64: gradient range extends from 0.0332 to 16.6553 and parity fails. No arbitrary-stack guarantee is carried forward. The prescribed depth-attenuated residual model is separately measured. |
| Residual pilot | `iterations/deep-pilot.json`, `.npz` | 200 independent trials per depth, width 64; gradient gates and parity pass. A pilot is not evidence for the required 10,000-trial gate. |
| Full width-64 study | `iterations/full-cpu-initial/metrics.json`, `deep-trials.npz` | 10,000 independent trials per depth. Gradient/parity gates pass, but maximum activation variance ratio is 2.746072. Low finite-width input variance makes this ratio noisy. Final width is 128, matching the TPU MXU, with unchanged per-trial thresholds. |
| Concurrent development audit | `iterations/full-cpu-initial.log` | Source was edited while the preliminary full study ran. The verifier correctly rejects it as `FAIL_SOURCE_CHANGED_DURING_RUN`; it cannot supply final provenance even for passing subsets. |
| Evidence validation tests | `iterations/pytest-records.log` | 33 tests pass, including rejection of missing, stale, and incomplete hardware evidence. |
| TPU availability | `cluster.json`, `tpu/availability.log` when present | Existing Cloud TPU v4-32 is 16 physical chips on four hosts. A separate live cauchylift training pipeline owns the devices. Leave that job running and recheck actual device holders before launching. CPU results do not establish hardware parity. |

## Reproduction

Use the complete commands in `REPRODUCE.md`. Each JSON includes its seed,
configuration, thresholds, and measurements. The full-study records include
source hashes and an environment fingerprint. The historical low-scale variance
artifact includes a minimal independent NumPy reproduction. The initial boundary
failure log includes the complete failing inputs and traceback.

## Scientific adjustments

`phases/phase1.md` §8 versions the corrected variance gate. `theory.md` corrects
both inflections, the negative-tail evaluation, AVN centered variance and radial
geometry, and the precise operation count/coupling scope. `DEPENDENCIES.md`
records the audit of consumers; no later-phase implementation exists yet.
