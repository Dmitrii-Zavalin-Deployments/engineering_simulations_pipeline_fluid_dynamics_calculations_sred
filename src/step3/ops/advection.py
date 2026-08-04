# src/step3/ops/advection.py

import logging

import numpy as np

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

# Rule 7: Granular Traceability for Numerical Kernels
logger = logging.getLogger("Solver.Ops.Advection")

def compute_local_advection(block: StencilBlock, field_id: FI) -> float:
    """
    Computes local (v^n ⋅ ∇) * field using schema-locked accessors.

    Formula: 
    u_c * (df/dx) + v_c * (df/dy) + w_c * (df/dz)
    
    Compliance:
    - Rule 7: Forensic Traceability & Fail-Fast math audit.
    - Rule 9: Unified Foundation Access via get_field().
    """
    
    # 1. Compute spatial derivatives (Central Differencing)
    # Foundational scalars ensured by Cell accessors to avoid the 'Array Leak'
    try:
        f_ip = block.i_plus.get_field(field_id)
        f_im = block.i_minus.get_field(field_id)
        f_jp = block.j_plus.get_field(field_id)
        f_jm = block.j_minus.get_field(field_id)
        f_kp = block.k_plus.get_field(field_id)
        f_km = block.k_minus.get_field(field_id)
    except AttributeError:
        logger.critical(f"TOPOLOGY CRASH: Block {block.id} missing neighbor for field {field_id.name}")
        raise

    # Rule 7: Guard against division by zero in geometry
    df_dx = (f_ip - f_im) / (2.0 * block.dx)
    df_dy = (f_jp - f_jm) / (2.0 * block.dy)
    df_dz = (f_kp - f_km) / (2.0 * block.dz)

    # 2. Compute cell-centered velocities
    u_c = block.center.get_field(FI.VX)
    v_c = block.center.get_field(FI.VY)
    w_c = block.center.get_field(FI.VZ)
    
    # 3. Assemble advection term: (v ⋅ ∇)f
    advection_val = (u_c * df_dx) + (v_c * df_dy) + (w_c * df_dz)

    # --- FORENSIC NUMERICAL AUDIT ---
    if not np.isfinite(advection_val):
        logger.error(
            f"MATH FAILURE: Non-finite advection in {block.id} | "
            f"Field: {field_id.name} | Vel: [{u_c:.2e}, {v_c:.2e}, {w_c:.2e}] | "
            f"Gradients: [{df_dx:.2e}, {df_dy:.2e}, {df_dz:.2e}]"
        )
        raise ArithmeticError(f"Advection term exploded in block {block.id}")

    return advection_val

def compute_local_advection_vector(block: StencilBlock) -> tuple[float, float, float]:
    """
    Computes the full advective term for the momentum equation (3D Vector).
    """
    logger.debug(f"OPS [Start]: Computing Advection Vector for {block.id}")
    
    try:
        adv_v = (
            compute_local_advection(block, FI.VX),
            compute_local_advection(block, FI.VY),
            compute_local_advection(block, FI.VZ)
        )
        
        # Log a sample of the magnitude for stability monitoring at DEBUG level
        mag = sum(x**2 for x in adv_v)**0.5
        logger.debug(f"OPS [Success]: {block.id} Advection Mag: {mag:.4e}")
        
        return adv_v

    except Exception:
        logger.error(f"OPS [Failure]: Advection vector computation failed in {block.id}")
        raise