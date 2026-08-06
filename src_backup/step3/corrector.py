# src/step3/corrector.py

import logging

import numpy as np

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.ops.gradient import compute_local_gradient_p
from src.step3.ops.scaling import get_dt_over_rho

logger = logging.getLogger("Solver.Corrector")

def apply_local_velocity_correction(block: StencilBlock) -> None:
    """
    Projects the intermediate velocity field v* onto a divergence-free space.
    Formula: v^{n+1} = v^* - (dt/rho) * grad(p^{n+1})
    """
    # 1. Compute the pressure gradient at p^{n+1} (FI.P_NEXT)
    grad_p = compute_local_gradient_p(block, field_id=FI.P_NEXT)
    
    # 2. Scaling factor (dt/rho)
    scaling = get_dt_over_rho(block)
    
    # 3. Retrieve intermediate star-velocity
    v_star = (
        block.center.get_field(FI.VX_STAR),
        block.center.get_field(FI.VY_STAR),
        block.center.get_field(FI.VZ_STAR)
    )
    
    # 4. Calculate new velocities
    v_new = [
        v_star[0] - (scaling * grad_p[0]),
        v_star[1] - (scaling * grad_p[1]),
        v_star[2] - (scaling * grad_p[2])
    ]
    
    # --- RULE 7: FAIL-FAST MATH CHECK ---
    if not np.isfinite(v_new).all():
        logger.error(f"CORRECTOR CRITICAL: Non-finite velocity in block {block.id}")
        raise ArithmeticError(f"Instability detected during velocity correction at block {block.id}")

    # 5. Apply velocity correction in-place to the STAR buffer
    block.center.set_field(FI.VX_STAR, v_new[0])
    block.center.set_field(FI.VY_STAR, v_new[1])
    block.center.set_field(FI.VZ_STAR, v_new[2])

    logger.debug(f"CORRECT [Success]: Block {block.id} updated with corrected velocities.")