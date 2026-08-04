# src/step3/ops/divergence.py

import logging

import numpy as np

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

# Rule 7: Granular Traceability for Numerical Kernels
logger = logging.getLogger("Solver.Ops.Divergence")

def compute_local_divergence_v_star(block: StencilBlock) -> float:
    """
    Computes the local scalar divergence ∇ ⋅ v* for the Pressure Poisson Equation.
    
    Formula:
    ∂u*/∂x + ∂v*/∂y + ∂w*/∂z
    
    Compliance:
    - Rule 7: Forensic Traceability & Fail-Fast math audit.
    - Rule 9: Unified Foundation Access via Sovereign Scalars.
    """
    
    # 1. Access intermediate velocity components (FI.VX_STAR, etc.)
    # The Cell foundation now ensures these are returned as native floats.
    try:
        u_ip = block.i_plus.get_field(FI.VX_STAR)
        u_im = block.i_minus.get_field(FI.VX_STAR)
        
        v_jp = block.j_plus.get_field(FI.VY_STAR)
        v_jm = block.j_minus.get_field(FI.VY_STAR)
        
        w_kp = block.k_plus.get_field(FI.VZ_STAR)
        w_km = block.k_minus.get_field(FI.VZ_STAR)
    except AttributeError as e:
        logger.critical(f"TOPOLOGY CRASH: Block {block.id} missing neighbors for Divergence.")
        raise

    # 2. Central difference: ∂u*/∂x + ∂v*/∂y + ∂w*/∂z
    # Rule 7: Defensive geometry audit
    try:
        div_x = (u_ip - u_im) / (2.0 * block.dx)
        div_y = (v_jp - v_jm) / (2.0 * block.dy)
        div_z = (w_kp - w_km) / (2.0 * block.dz)
    except ZeroDivisionError as e:
        logger.critical(f"GEOMETRY CRASH: Block {block.id} has invalid dimensions (dx={block.dx})")
        raise
    
    divergence_val = div_x + div_y + div_z

    # --- FORENSIC NUMERICAL AUDIT ---
    # Rule 7: Detect PPE Poisoning before it starts (Critical for SOR stability)
    if not np.isfinite(divergence_val):
        logger.error(
            f"MATH FAILURE: Non-finite divergence in {block.id} | "
            f"Components: [dx:{div_x:.2e}, dy:{div_y:.2e}, dz:{div_z:.2e}] | "
            f"Result: {divergence_val:.4e}"
        )
        raise ArithmeticError(f"Divergence exploded in block {block.id}. PPE source term is poisoned.")

    return divergence_val