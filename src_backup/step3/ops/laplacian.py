# src/step3/ops/laplacian.py

import logging

import numpy as np

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

# Rule 7: Granular Traceability for Numerical Diffusion Kernels
logger = logging.getLogger("Solver.Ops.Laplacian")

def compute_local_laplacian(block: StencilBlock, field_id: FI) -> float:
    """
    Computes the discrete Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
    
    Compliance:
    - Rule 7: Forensic Traceability & Fail-Fast math audit.
    - Rule 8: Centralized logic prevents drift between velocity and pressure ops.
    - Rule 9: Unified Foundation Access via Sovereign Scalars (Native Floats).
    """
    
    # 1. Access center and neighbors via Foundation schema
    # The Cell foundation now ensures these are returned as native floats.
    try:
        f_c = block.center.get_field(field_id)
        
        f_ip = block.i_plus.get_field(field_id)
        f_im = block.i_minus.get_field(field_id)
        
        f_jp = block.j_plus.get_field(field_id)
        f_jm = block.j_minus.get_field(field_id)
        
        f_kp = block.k_plus.get_field(field_id)
        f_km = block.k_minus.get_field(field_id)
    except AttributeError:
        logger.critical(f"TOPOLOGY CRASH: Block {block.id} missing neighbors for Laplacian of {field_id.name}.")
        raise

    # 2. Geometry Setup (Rule 4: SSoT from block)
    if block.dx <= 0 or block.dy <= 0 or block.dz <= 0:
        logger.critical(f"GEOMETRY CRASH: Block {block.id} has non-positive dimensions.")
        raise ZeroDivisionError(f"Invalid geometry in block {block.id}")

    dx2, dy2, dz2 = block.dx**2, block.dy**2, block.dz**2

    # 3. Discrete Laplacian Calculation (Standard 7-point stencil)
    # Pure scalar math - no NumPy object creation overhead.
    term_x = (f_ip - 2.0 * f_c + f_im) / dx2
    term_y = (f_jp - 2.0 * f_c + f_jm) / dy2
    term_z = (f_kp - 2.0 * f_c + f_km) / dz2 
    
    lap_val = term_x + term_y + term_z

    # --- FORENSIC NUMERICAL AUDIT ---
    # Rule 7: Catch instability before it propagates through the SOR step.
    if not np.isfinite(lap_val):
        logger.error(
            f"MATH FAILURE: Non-finite Laplacian in {block.id} | "
            f"Field: {field_id.name} | Center Val: {f_c:.2e} | "
            f"Terms [X:{term_x:.2e}, Y:{term_y:.2e}, Z:{term_z:.2e}]"
        )
        raise ArithmeticError(f"Laplacian exploded in block {block.id}")

    return lap_val

def compute_local_laplacian_v_n(block: StencilBlock) -> tuple[float, float, float]:
    """Computes Laplacian for primary velocity components (v^n)."""
    logger.debug(f"OPS [Start]: Computing Laplacian Vector (v^n) for {block.id}")
    try:
        return (
            compute_local_laplacian(block, FI.VX),
            compute_local_laplacian(block, FI.VY),
            compute_local_laplacian(block, FI.VZ)
        )
    except Exception:
        logger.error(f"OPS [Failure]: Laplacian vector computation failed for {block.id}")
        raise

def compute_local_laplacian_p_next(block: StencilBlock) -> float:
    """Computes Laplacian for trial pressure p^{n+1}."""
    logger.debug(f"OPS [Start]: Computing Laplacian P_NEXT for {block.id}")
    return compute_local_laplacian(block, FI.P_NEXT)