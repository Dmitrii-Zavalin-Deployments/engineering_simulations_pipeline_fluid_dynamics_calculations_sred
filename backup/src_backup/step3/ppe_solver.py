# src/step3/ppe_solver.py

import logging

import numpy as np

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.ops.divergence import compute_local_divergence_v_star
from src.step3.ops.scaling import get_rho_over_dt

logger = logging.getLogger("Solver.PPE")


def solve_pressure_poisson_step(
    block: StencilBlock, divergence_threshold: float, omega: float
) -> float:
  """Consolidated PPE Solver using SOR iteration with Fail-Fast Math.

  Compliance:
  - Rule 7: Immediate math audit (Atomic Verification).
  - Rule 9: Performs in-place updates via schema-locked accessors.
  """
  # 1. Geometry Setup (SSoT from StencilBlock)
  dx2, dy2, dz2 = block.dx**2, block.dy**2, block.dz**2
  stencil_denom = 2.0 * (1.0 / dx2 + 1.0 / dy2 + 1.0 / dz2)

  # 2. Compute Rhie-Chow Stabilization
  # Access center once to minimize cache misses/Foundation calls
  p_c = block.center.get_field(FI.P)

  lap_p_n = (
      (
          block.i_plus.get_field(FI.P)
          - 2.0 * p_c
          + block.i_minus.get_field(FI.P)
      )
      / dx2
      + (
          block.j_plus.get_field(FI.P)
          - 2.0 * p_c
          + block.j_minus.get_field(FI.P)
      )
      / dy2
      + (
          block.k_plus.get_field(FI.P)
          - 2.0 * p_c
          + block.k_minus.get_field(FI.P)
      )
      / dz2
  )

  # Stabilization weights the pressure correction by the current time-step
  rhie_chow_term = block.dt * lap_p_n

  # 3. Compute RHS (Source Term)
  div_v_star = compute_local_divergence_v_star(block)

  # --- RULE 7: FAIL-FAST MATH AUDIT (SOURCE) ---
  if not np.isfinite(div_v_star):
    logger.error(f"PPE MATH ERROR: Non-finite divergence in block {block.id}")
    raise ArithmeticError(f"NaN detected in divergence source term: {div_v_star}")

  rho_over_dt = get_rho_over_dt(block)
  rhs = rho_over_dt * (div_v_star - rhie_chow_term)

  # 4. SOR Update Prep (Phase 2: Solve & Correct)
  # sum_neighbors uses P_NEXT to represent the most current trial values in the SOR sweep
  sum_neighbors = (
      (
          block.i_plus.get_field(FI.P_NEXT)
          + block.i_minus.get_field(FI.P_NEXT)
      )
      / dx2
      + (
          block.j_plus.get_field(FI.P_NEXT)
          + block.j_minus.get_field(FI.P_NEXT)
      )
      / dy2
      + (
          block.k_plus.get_field(FI.P_NEXT)
          + block.k_minus.get_field(FI.P_NEXT)
      )
      / dz2
  )

  p_old = block.center.get_field(FI.P_NEXT)

  # --- RULE 7: PRE-UPDATE AUDIT ---
  # Catch non-finite values or extreme divergence before the SOR calculation
  # pollutes the entire grid. Strict access policy: no default fallbacks.
  if not np.isfinite(p_old) or abs(p_old) > divergence_threshold:
    logger.error(
        f"\n==================================================\n"
        f"🔥 PPE CRITICAL: Poisoned p_old in block {block.id}\n"
        f"--------------------------------------------------\n"
        f"  Val (p_old):           {p_old:.2e} (Threshold:"
        f" {divergence_threshold:.2e})\n"
        f"  Center Pressure (p_c): {p_c:.2e}\n"
        f"  Divergence (div_v*):   {div_v_star:.2e}\n"
        f"  Rhie-Chow Term:        {rhie_chow_term:.2e}\n"
        f"  RHS Source Term:       {rhs:.2e}\n"
        f"  Sum Neighbors:         {sum_neighbors:.2e}\n"
        f"  Stencil Denominator:   {stencil_denom:.2e}\n"
        f"  Grid Spacing (dx,dy,dz): {block.dx:.2e}, {block.dy:.2e},"
        f" {block.dz:.2e}\n"
        f"--------------------------------------------------\n"
        f"  [TOPOLOGY & NEIGHBOR PROVENANCE]\n"
        f"  i_minus index: {block.i_minus.index} | p_next: {block.i_minus.get_field(FI.P_NEXT):.2e}\n"
        f"  i_plus  index: {block.i_plus.index}  | p_next: {block.i_plus.get_field(FI.P_NEXT):.2e}\n"
        f"  j_minus index: {block.j_minus.index} | p_next: {block.j_minus.get_field(FI.P_NEXT):.2e}\n"
        f"  j_plus  index: {block.j_plus.index}  | p_next: {block.j_plus.get_field(FI.P_NEXT):.2e}\n"
        f"  k_minus index: {block.k_minus.index} | p_next: {block.k_minus.get_field(FI.P_NEXT):.2e}\n"
        f"  k_plus  index: {block.k_plus.index}  | p_next: {block.k_plus.get_field(FI.P_NEXT):.2e}\n"
        f"=================================================="
    )
    raise ArithmeticError(f"Poisoned Pressure Trial detected at {block.id}")

  # 5. Calculate Trial Pressure via SOR Formula
  # p_new = (1-w)*p_old + w*(Source + Neighbors)/Denom
  p_new = (1.0 - omega) * p_old + (omega / stencil_denom) * (
      sum_neighbors - rhs
  )

  # --- RULE 7: POST-UPDATE AUDIT ---
  if not np.isfinite(p_new):
    logger.error(
        f"PPE MATH ERROR: Non-finite p_new in block {block.id} | Result:"
        f" {p_new}"
    )
    raise ArithmeticError("Non-finite pressure generated in SOR step")

  # Rule 0: Robust delta calculation for convergence monitoring
  delta = float(abs(p_new - p_old))

  # 6. Direct write-back to foundation (Sovereign Scalar commit)
  block.center.set_field(FI.P_NEXT, p_new)

  return delta