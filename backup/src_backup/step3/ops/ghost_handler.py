# src/step3/ops/ghost_handler.py

import logging

from src.common.stencil_block import StencilBlock

logger = logging.getLogger("Solver.GhostHandler")

def sync_ghost_trial_buffers(block: StencilBlock) -> None:
    """
    Rule 7 & 9: Direct Buffer Alignment for Ghost Cells.
    
    Ensures Trial buffers (STAR, NEXT) are synchronized with foundation values
    (VX, VY, VZ, P) directly in the NumPy memory space.
    
    This is critical for boundary stability: it ensures that non-fluid cells
    provide a consistent "mirror" or "static" value for the stencil math.
    
    Constraint: No temporary arrays, dicts, or heap reallocations.
    """
    # Rule 9: Accessing the center cell's logic-pointer from the StencilBlock
    cell = block.center
    
    try:
        # Rule 9: Sovereign Scalar Access via @property 
        # These operations read a native float from the foundation and 
        # write it directly back to the trial buffer index.
        cell.vx_star = cell.vx
        cell.vy_star = cell.vy
        cell.vz_star = cell.vz
        
        # Sync Pressure Trial Buffer (P -> P_NEXT)
        cell.p_next = cell.p

        # Rule 7: High-resolution debug log for traceability in the field
        logger.debug(f"GHOST SYNC [Success]: Block {block.id} trial buffers aligned.")
        
    except AttributeError as e:
        # Rule 5: Explicit Error on structural failure (Foundation Integrity)
        logger.critical(
            f"CONTRACT VIOLATION: Block {block.id} center lacks Rule 9 field properties. "
            f"Error: {e}"
        )
        raise RuntimeError("Ghost sync failed: Cell foundation is malformed.") from e