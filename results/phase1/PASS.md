# Phase 1 PASS — gate version 2

Reverified 2026-09-18 on the required four-host, 16-chip TPU v4 slice. All CPU, formal, numerical, purity, statistical, and hardware gates pass.

## Gate inventory

| Gate | Outcome and direct evidence |
| --- | --- |
| Lean certificates | Clean root build; no warnings or project placeholders: [lean-build.log](lean-build.log), `metrics.json:formal` |
| Unit and evidence validation | 128 repository tests passed: [pytest.log](pytest.log) |
| Algebraic purity | Zero forbidden source/graph operations; cached ALU/AVN VJPs contain no division or rsqrt: `metrics.json:purity` |
| Reflection / derivative / inflection | Errors 0 / 4.440892098500626e-16 / 0: `metrics.json:numerical` |
| ALU Lipschitz | 1.0443310539518174 ≤ 1.05: `metrics.json:numerical` |
| AVN moment and variance v2 | 1,000,000 samples at each of seven scales on CPU and TPU; exact regularized identities and epsilon-zero unit-variance CIs pass: `metrics.json:monte_carlo`, `tpu/metrics.json:monte_carlo` |
| Deep gradient flow and variance | 10,000 independent trials at each depth 8/16/24/32, width 128, on CPU and TPU. All gradient ratios in [0.2,5], all depth-32 activation ratios in [0.5,2], all ALU/GELU variance deltas ≤5%: `metrics.json:deep`, `tpu/metrics.json:deep`; raw [CPU](deep-trials.npz) and [TPU](tpu/deep-trials.npz) arrays |
| TPU float32/BF16 parity | All 16 forward/VJP configurations pass including scale 1e15: `tpu/metrics.json:parity` |
| Synchronized TPU throughput | All ALU/GELU ≥0.90 and AVN/RMSNorm ≥0.95 forward and forward+backward gates pass: `tpu/metrics.json:benchmarks`, [raw latencies](tpu/latencies.json) |
| AVN parameter memory | Zero bytes versus 16,384 RMSNorm / 32,768 LayerNorm bytes at width 4096: `tpu/metrics.json:benchmarks` |

## TPU throughput and uncertainty

100 randomized-order repetitions after 10 warmups; global shape 4096×4096. Each latency ends in `block_until_ready`, takes the slowest host, and excludes compilation, transfers, and barriers. Ratios use medians; intervals below describe latency means.

| Dtype | Operation | Latency µs, mean ± SEM | 95% CI µs | Baseline / algebraic throughput |
| --- | --- | --- | --- | --- |
| float32 | alu_forward | 363.07 ± 16.52 | [330.30, 395.84] | 1.0633 |
| float32 | avn_forward | 463.85 ± 115.80 | [234.08, 693.61] | 1.0362 |
| float32 | alu_forward_backward | 448.98 ± 2.63 | [443.77, 454.19] | 1.0424 |
| float32 | avn_forward_backward | 466.66 ± 7.66 | [451.46, 481.86] | 1.2220 |
| bfloat16 | alu_forward | 350.49 ± 2.67 | [345.19, 355.79] | 1.0356 |
| bfloat16 | avn_forward | 350.22 ± 4.01 | [342.27, 358.17] | 1.0230 |
| bfloat16 | alu_forward_backward | 437.53 ± 2.96 | [431.66, 443.39] | 1.0365 |
| bfloat16 | avn_forward_backward | 451.30 ± 9.92 | [431.61, 470.99] | 1.2209 |

## TPU residual study

| Depth | Gradient ratio mean ± SEM | 95% CI | Observed range | Variance delta vs GELU |
| --- | --- | --- | --- | --- |
| 8 | 1.019619 ± 0.001356 | [1.016960, 1.022278] | [0.585492, 1.814757] | 1.655% |
| 16 | 1.021745 ± 0.001384 | [1.019032, 1.024458] | [0.619460, 1.639585] | 1.797% |
| 24 | 1.017926 ± 0.001367 | [1.015246, 1.020606] | [0.604180, 1.802536] | 1.767% |
| 32 | 1.019519 ± 0.001366 | [1.016841, 1.022196] | [0.575426, 1.634833] | 1.821% |

CPU statistics and all AVN confidence intervals: [SUMMARY.md](SUMMARY.md). Complete baseline statistics and per-scale TPU intervals are in the JSON records. Independent vectors/trials are the statistical units; intervals are Student-t 95% intervals. Figures: [CPU PNG](verification.png), [CPU PDF](verification.pdf), [TPU PNG](tpu-verification.png), [TPU PDF](tpu-verification.pdf), all PNGs at 300 dpi.

## Reproduction from a fresh shell

### Setup and commands

## Fresh host setup and CPU/formal verification

Run from a fresh shell. A Python 3.10 environment and the committed Lean toolchain
are used. On Ubuntu, install `python3.10-venv` if `python3 -m venv` reports that
ensurepip is missing. Install elan using its official installer if absent.

```bash
git clone https://github.com/tasmaikeni13/algebraic-intelligence.git
cd algebraic-intelligence
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PATH="$HOME/.elan/bin:$PATH"
(cd formal && lake exe cache get && lake build)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_verify_primitives.py --cpu-only
```

`--cpu-only` runs all 10,000 trials at each of four depths and all million-sample
scale experiments. It returns success for the CPU subset and explicitly records
Phase 1 as incomplete until current TPU evidence is present. Without this flag,
missing hardware evidence makes the command exit nonzero.

## Four-host TPU verification

The existing node `my-tpu-v4` in `us-central2-b` is Cloud TPU **v4-32**: 16 physical
v4 chips, four hosts, physical topology 2×2×4. No new node is provisioned. The
launcher checks that devices are free and exits without stopping other jobs.
It bundles the committed source, makes isolated snapshot directories on all four
hosts, installs pinned dependencies, runs CPU tests on each host, then starts the
same JAX distributed program on every host.

```bash
.venv/bin/python scripts/launch_phase1_tpu.py --name my-tpu-v4 --zone us-central2-b
.venv/bin/python scripts/run_verify_primitives.py
```

The second command validates TPU source hashes, the complete device inventory,
float32/BF16 numerical parity, Monte Carlo identities, all 40,000 deep trials,
and ALU/GELU and AVN/RMSNorm forward and forward+backward throughput thresholds.
Timing includes `jax.block_until_ready`, excludes compilation/transfers, uses
10 warmups and 100 randomized-order repetitions, and reports the slowest host's
latency in each sample. RMSNorm/LayerNorm receive dynamic learnable scale arrays;
their backward timing includes parameter cotangents. Swish and LayerNorm are
additional measured comparison arms. Raw timing samples and trials are retained.

For another already-provisioned matching slice, supply its name and zone. The
program rejects other device counts or accelerator generations. It never counts
CPU or emulated-device timings as TPU measurements.

## Evidence and limits

See `metrics.json`, `STATUS.md`, `ITERATIONS.md`, and `DEPENDENCIES.md`. A Phase 1
PASS record is written only after every required CPU/formal and TPU gate passes
and the artifacts have been audited. All test-network claims specify width 128,
He initialization, residual attenuation, seeds, and sample counts. Phase 1 does
not establish trainability or performance for later transformer implementations.

## Revalidate preserved CPU samples after a metadata-only change

```bash
.venv/bin/python scripts/run_verify_primitives.py --reuse-cpu results/phase1/iterations/full-cpu-width128/metrics.json
```

The verifier rejects changed numerical dependencies or numerical package versions;
it always rebuilds Lean, runs the unit suite and purity audit, and checks current
TPU evidence. Omit `--reuse-cpu` to repeat every CPU sample. TPU output is collected
from all four hosts because cloud worker order need not equal JAX process order.

## Failures, repairs, theory, and dependencies

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

## Additional verification before the TPU deferral

- The four-host preflight refused to start while the devices were owned by an
  existing process; `iterations/tpu-launch-busy.log` and `tpu/availability.log`
  preserve the command failure. No process was stopped.
- An initial CPU smoke invocation inherited the hardware runner's TPU backend
  setting (`iterations/cpu-runner-smoke.log`). Explicitly setting JAX's backend
  configuration to CPU fixed the invocation. This was a test invocation issue;
  the hardware runner intentionally requests TPU execution.
- `iterations/cpu-runner-smoke-retry.log` exercises sharded primitive helpers and
  all forward/forward+backward benchmark call signatures in float32 and bfloat16
  on four emulated **CPU** devices. Its timings are not TPU gate evidence.
- On 2026-09-15 the user deferred TPU use until the following day and instructed
  publication of the work that can be completed now. No TPU job is queued or
  scheduled by this implementation. Run the documented launcher when available.

## Final CPU verification

`iterations/full-cpu-width128/metrics.json` records CPU_VERIFIED_TPU_PENDING. All CPU/formal gates pass for the complete 128-feature study; the source hashes match the committed snapshot. Its raw trial arrays and figures are preserved. The top-level copies and SUMMARY.md make this result easy to inspect. No failing historical artifact was overwritten.

## TPU execution, 2026-09-16

- `iterations/tpu-mesh-failure/` preserves the first distributed failure: the
  requested named mesh needed physical-axis splitting on this topology. Enabling
  `allow_split_physical_axes=True` repaired placement without changing kernels.
- `iterations/tpu-collection-failure/recovered/metrics.json` records the first
  complete hardware PASS at commit `108f929`. Cloud worker 1 was JAX process 0;
  the original worker-0-only download missed its output. The full record was
  recovered without changing it. This was an artifact-collection bug.
- The launcher now uses an empty measurement directory, fetches every host into
  a unique directory, and requires exactly one coordinator record. This prevents
  stale bundled results from being mistaken for a new measurement.
- `iterations/tpu-final-launch.log` and `tpu/` contain the final complete repeat
  after the launcher repair. The aggregate verifier checks source fingerprints,
  every hardware gate, fresh CPU unit tests, purity, and the complete Lean build.
- The CPU sample reuse path checks every numerical dependency and numerical
  package version; `metrics.json` preserves the original CPU measurement commit,
  dirty status, command, and hashes. Reusing those unchanged samples does not
  relabel them as a new measurement. A unit test rejects stale dependencies.


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


`theory.md` now states both ALU inflections, rationalized negative-tail evaluation, exact regularized second-moment/centered-variance identities, positive-scale gate coupling, and the restricted projection interpretation at epsilon=0. Phase 1 gate v2 and the historical failures remain visible in `phases/phase1.md` §8. Thresholds for hardware speed and gradient parity are unchanged.

Lean additions: `alu_inflection_iff`, `alu_negative_tail_identity`, `alu_derivative_bound`, `avn_positive_scale_invariance`, `avn_gate_coupling`, `avn_regularized_moment`, `avn_centered_variance`, `avn_radial_damping`. Exact assumptions and proof limits: [PROOF_COVERAGE.md](../../formal/PROOF_COVERAGE.md).

## Provenance

TPU measurement commit: `0ab241a312c7767980e0e8162824c4b3f06662a3`; dirty state: `False`. Aggregate audit commit: `0ab241a312c7767980e0e8162824c4b3f06662a3`; dirty state: `True` (generated evidence/docs). Original CPU samples retain their own source hashes, command, commit and dirty state in `metrics.json:cpu_measurement_provenance`; their numerical dependencies are byte-identical. The aggregate audit does not claim a new sampling run.

Cluster: GCloud project `projectalgebraicai`, node `my-tpu-v4`, zone `us-central2-b`, accelerator type `v4-32` (16 physical TPU v4 chips), four hosts, 2×2×4 named data/fsdp/model mesh with physical-axis splitting. JAX/jaxlib 0.6.2, libtpu 0.0.17. All 16 device IDs, host indices and physical coordinates are in `tpu/metrics.json:devices`; cloud inventory is [cluster.json](cluster.json). Actual coordinator host and versions are in each environment record.

## Limits

The deep result applies to the measured normalized residual network, width 128 and initialization distribution, not arbitrary networks or trained transformers. Default AVN does not promise unit centered variance. Reduced-precision guarantees require squared accumulations to stay finite. BF16 uses float32 intermediates. Throughput is for the recorded shapes and hardware and includes host dispatch overhead; later attention/fused kernels need their own benchmarks. Real-arithmetic Lean proofs do not certify machine rounding or statistical generalization. Later phases are not certified by this PASS.
