import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

namespace AlgebraicTheory

theorem kernel_reciprocal_identity (x s : ℝ) (hs : s^2 = x^2 + 1) :
    (s + x) * (s - x) = 1 := by
  calc
    (s + x) * (s - x) = s^2 - x^2 := by ring
    _ = (x^2 + 1) - x^2 := by rw [hs]
    _ = 1 := by ring

theorem kernel_power_eight_identity (rho : ℝ) :
    (((rho * rho) * (rho * rho)) * ((rho * rho) * (rho * rho))) = rho^8 := by
  ring

end AlgebraicTheory
