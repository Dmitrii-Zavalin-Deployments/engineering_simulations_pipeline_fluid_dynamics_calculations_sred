# src/step3/boundaries/applier.py

import logging

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

logger = logging.getLogger("Solver.Boundaries")

# Rule 9 Compliance: Trial Field Redirection
# Maps logical physics keys (u, v, w, p) to specific Trial/Next buffers.
# This ensures Step 3 iterations never pollute the n-step base fields.
BC_FIELD_MAP = {
    "u": FI.VX_STAR,
    "v": FI.VY_STAR,
    "w": FI.VZ_STAR,
    "p": FI.P_NEXT
}

def apply_boundary_values(block: StencilBlock, rule: dict) -> None:
    """
    Applies boundary values to the StencilBlock Trial Fields.
    Operates directly on Foundation memory via the StencilBlock interface.
    
    Compliance:
    - Rule 7: Fail-Fast logging for contract violations.
    - Rule 9: In-place update via Sovereign Scalar Accessors (Center-Only).
    """
    values = rule.get("values")
    location = rule.get("location")
    boundary_type = rule.get("type")
    
    # 1. Validation: Ensure the rule carries the required payload
    if values is None or location is None or boundary_type is None:
        logger.error(f"STRATEGIC FAILURE: Missing fields at location={location}")
        raise ValueError(f"Boundary rule missing critical fields at location={location}")

    # 2. Vectorized Field Application
    for key, value in values.items():
        field_id = BC_FIELD_MAP.get(key)
        
        if field_id is not None:
            # Rule 9: Sovereign Accessor. Writing directly to the center cell 
            # of the stencil block to enforce the boundary value.
            block.center.set_field(field_id, value)
            
            # Rule 7: Promotion to INFO level to satisfy Quality Gate Traceability.
            # This ensures the MMS Audit can verify the 'poisoned' cell was corrected.
            logger.debug(f'Mapping u to Field {field_id}')
            logger.info(
                f"APPLY [Success]: Block {block.id} | Location: {location} | "
                f"Field {field_id.name} set to {float(value):.4e}"
            )
            
        else:
            # Rule 7: Fail-Fast on configuration drift (e.g., trying to set temperature in NS)
            logger.error(f"CONTRACT VIOLATION: Unsupported key '{key}' at {location}")
            raise KeyError(f"Unsupported boundary key '{key}' at {location}")