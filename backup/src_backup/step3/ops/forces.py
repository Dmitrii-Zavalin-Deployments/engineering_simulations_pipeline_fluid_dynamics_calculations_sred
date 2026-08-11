# src/step3/ops/forces.py

import logging

import numpy as np

from src.common.stencil_block import StencilBlock

# Rule 7: Granular Traceability for Momentum Drivers
logger = logging.getLogger("Solver.Ops.Forces")

def get_local_body_force(block: StencilBlock) -> tuple[float, float, float]:
    """
    Returns the body force vector (Fx, Fy, Fz) for the current stencil block.
    
    Compliance:
    - Rule 7: Fail-Fast validation. Ensures input forces are finite before 
      they enter the momentum predictor calculation.
    - Rule 4 (SSoT): Forces are retrieved from the block's immutable properties.
    - Rule 8: Contract Guard. Validates that f_vals is a valid 3-component vector.
    """
    
    # 1. Contract & Topology Guard (Rule 8)
    try:
        forces = block.f_vals
        
        # Ensure we have exactly 3 components (X, Y, Z)
        if len(forces) != 3:
            logger.critical(
                f"CONTRACT VIOLATION: Block {block.id} has invalid force vector length: {len(forces)}. "
                "Expected 3 components (Fx, Fy, Fz)."
            )
            raise ValueError(f"Invalid body force vector in block {block.id}")
            
    except AttributeError as e:
        logger.critical(
            f"TOPOLOGY CRASH: Block {block.id} is missing force definitions (f_vals). "
            "Check Step 2 Stencil Assembler logic."
        )
        # Compliance (B904): Preserve traceback
        raise AttributeError(f"Body force metadata missing in block {block.id}") from e

    # 2. Forensic Numerical Audit (Rule 7)
    # We check for finiteness here to prevent 'NaN' from entering the VX_STAR calculation
    if not np.isfinite(forces).all():
        logger.error(
            f"MATH FAILURE: Non-finite body forces detected in {block.id} | "
            f"F_vals: [{forces[0]:.2e}, {forces[1]:.2e}, {forces[2]:.2e}]"
        )
        raise ArithmeticError(f"Body force is non-finite in block {block.id}")

    # 3. Successful Execution Trace
    logger.debug(
        f"OPS [Success]: Body force retrieved for {block.id} | "
        f"G-Vector: {forces}"
    )
    
    return forces