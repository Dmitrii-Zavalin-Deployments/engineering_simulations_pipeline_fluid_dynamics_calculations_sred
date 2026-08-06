# src/step3/ops/gradient.py

import logging

import numpy as np

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

# Rule 7: Granular Traceability for Pressure-Poisson Coupling
logger = logging.getLogger("Solver.Ops.Gradient")

def compute_local_gradient_p(block: StencilBlock, field_id: FI = FI.P) -> tuple[float, float, float]:
    """
    Computes the discrete pressure gradient: ∇p = (∂p/∂x, ∂p/∂y, ∂p/∂z)
    
    Compliance:
    - Rule 7: Fail-Fast math audit. Raises ArithmeticError on non-finite results.
    - Rule 8: Topology Guard. Ensures neighbor presence before buffer access.
    - Rule 9: Unified Foundation Access via Sovereign Scalars (Native Floats).
    """
    
    # 1. Topology & Field Access Guard (Rule 8)
    try:
        # Central difference requires both neighbors on each axis.
        # The Cell foundation now ensures these are returned as native floats.
        p_im = block.i_minus.get_field(field_id)
        p_ip = block.i_plus.get_field(field_id)
        
        p_jm = block.j_minus.get_field(field_id)
        p_jp = block.j_plus.get_field(field_id)
        
        p_km = block.k_minus.get_field(field_id)
        p_kp = block.k_plus.get_field(field_id)
        
    except AttributeError as e:
        # Rule 7: Log the topology break with CRITICAL severity
        logger.critical(
            f"TOPOLOGY CRASH: Block {block.id} missing neighbors for Gradient of {field_id.name}. "
            "Check boundary condition synchronization."
        )
        raise AttributeError(f"Incomplete stencil for gradient in block {block.id}") from e

    # 2. Geometry Guard (Rule 7)
    if block.dx <= 0 or block.dy <= 0 or block.dz <= 0:
        logger.critical(
            f"GEOMETRY CRASH: Block {block.id} has invalid dimensions: "
            f"dx={block.dx}, dy={block.dy}, dz={block.dz}"
        )
        raise ZeroDivisionError(f"Invalid grid spacing in block {block.id}")

    # 3. Numerical Calculation (2nd Order Central Difference)
    # Pure scalar math - no NumPy overhead here.
    grad_x = (p_ip - p_im) / (2.0 * block.dx)
    grad_y = (p_jp - p_jm) / (2.0 * block.dy)
    grad_z = (p_kp - p_km) / (2.0 * block.dz)
    
    grad = (grad_x, grad_y, grad_z)

    # 4. Forensic Numerical Audit (Rule 7)
    # We use .all() because grad is a tuple of components.
    if not np.isfinite(grad).all():
        logger.error(
            f"MATH FAILURE: Gradient exploded in {block.id} | "
            f"Field: {field_id.name} | "
            f"Grad components: [{grad_x:.2e}, {grad_y:.2e}, {grad_z:.2e}]"
        )
        raise ArithmeticError(f"Pressure gradient is non-finite in block {block.id}")

    logger.debug(f"OPS [Success]: Gradient calculated for {block.id} | Field: {field_id.name}")
    return grad