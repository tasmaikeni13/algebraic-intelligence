import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

namespace AlgebraicTheory

theorem gate_reflection_identity (u : ℝ) :
    (1 / 2 : ℝ) * (1 + u) + (1 / 2 : ℝ) * (1 - u) = 1 := by
  ring

theorem alu_cache_invertibility (x s u : ℝ) (hs : s^2 = x^2 + 1) (hu : u * s = x) :
    (1 - u^2) * s^2 = 1 := by
  calc
    (1 - u^2) * s^2 = s^2 - (u * s)^2 := by ring
    _ = (x^2 + 1) - x^2 := by rw [hs, hu]
    _ = 1 := by ring

theorem alu_polynomial_backward_identity (u : ℝ) :
    (1 / 2 : ℝ) * (1 + u) + (1 / 2 : ℝ) * (u - u^3) =
    (1 / 2 : ℝ) * (1 + 2 * u - u^3) := by
  ring

theorem alu_inflection_identity (u : ℝ) (hu : 3 * u^2 = 2) :
    2 - 3 * u^2 = 0 := by
  linarith

theorem alu_inflection_x_to_u (x_sq s_sq u_sq : ℝ)
    (hx : x_sq = 2) (hs : s_sq = x_sq + 1) (hu : u_sq * s_sq = x_sq) :
    u_sq = 2 / 3 := by
  have hs_val : s_sq = 3 := by linarith
  rw [hs_val] at hu
  linarith

end AlgebraicTheory
