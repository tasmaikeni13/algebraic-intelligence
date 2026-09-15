# Lean verification

The committed toolchain is **Lean 4.34.0-rc2**, with Mathlib pinned by
`lake-manifest.json` to `7974e751bece493b6ff508039423ca9fa2452fa8`.
The old prose referring to Lean 4.16.0 did not match the repository toolchain.

```bash
export PATH="$HOME/.elan/bin:$PATH"
cd formal
lake exe cache get
lake build
```

`lake` was installed at `/home/tas_ken_rt25/.elan/bin/lake` on this machine;
the historical `/root/.elan/bin/lake` path is not portable. The same committed
project and compiler are used. See [proof coverage](PROOF_COVERAGE.md) and the
[Phase 1 build log](../results/phase1/lean-build.log) for direct evidence.

## Phase 1 modules

- `AlgebraicTheory/Gate.lean`: reflection symmetry, cached derivative polynomial
  identity, cache invertibility, both inflection coordinates in squared form,
  the rationalized negative tail, and a rational derivative bound on [-1,1].
- `AlgebraicTheory/Variance.lean`: bounded norm algebra, positive scaling through
  the actual real square root, gate coupling, regularized moment and centered
  variance identities, and the nonzero radial damping at positive epsilon.

The root build also compiles the pre-existing Kernel, Cayley, Loss, and Curvature
modules. Compiling these algebraic lemmas does not complete Phases 2–10.
No project proof contains `sorry`, `admit`, or a declared `axiom`.
