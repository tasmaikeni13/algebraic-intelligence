# Phase 1: verification in progress

The 33 unit tests and Lean build pass. The original unit-variance gate was
mathematically inconsistent with regularized, uncentered AVN; its versioned
replacement and historical failures are recorded in `ITERATIONS.md`.

The full 128-feature CPU study is running. TPU verification is pending because
an existing training pipeline owns the 16-chip slice. Phase 1 has not passed.

See [reproduction](REPRODUCE.md), [failures and repairs](ITERATIONS.md),
[dependency audit](DEPENDENCIES.md), and [cluster fingerprint](cluster.json).
