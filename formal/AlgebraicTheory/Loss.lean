import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace AlgebraicTheory

theorem pearson_divergence_expansion (y p : ℝ) (hp : p ≠ 0) :
    (y - p)^2 / p = y^2 / p - 2 * y + p := by
  field_simp [hp]
  ring

theorem oace_power_chain (z : ℝ) :
    (((z^2)^2)^2) = z^8 := by
  ring

end AlgebraicTheory
