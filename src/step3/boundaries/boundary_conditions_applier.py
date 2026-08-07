# src/step3/boundaries/boundary_conditions_applier.py

import logging
from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

logger = logging.getLogger("Solver.Boundaries")

# Rule 9 Compliance: Redirection to Trial/Next buffers
BC_FIELD_MAP = {
    "u": FI.VX_STAR,
    "v": FI.VY_STAR,
    "w": FI.VZ_STAR,
    "p": FI.P_NEXT,
}

def apply_boundary_conditions(
    block: StencilBlock,
    state_bc_manager: object,
    state_grid: object,
    domain_cfg: dict
) -> None:
    """
    Applies physical boundary conditions strictly to boundary cells (mask == -1).
    Does NOT handle halo/ghost synchronization (is_ghost), which is managed by ghost_handler.
    """
    mask = getattr(block.center, "mask", 1)
    is_boundary = getattr(block.center, "is_boundary", False)

    # 1. Strict Boundary Filter: Only execute where mask == -1
    if mask != -1 and not is_boundary:
        return

    boundary_cfg = state_bc_manager.to_dict() if hasattr(state_bc_manager, "to_dict") else state_bc_manager
    location = _detect_boundary_location(block)

    # 2. Resolve target boundary values from sovereign config
    rule_values = _resolve_boundary_values(boundary_cfg, domain_cfg, location)

    # 3. Apply field values directly to trial buffers
    for key, val in rule_values.items():
        field_id = BC_FIELD_MAP.get(key)
        if field_id is not None:
            block.center.set_field(field_id, val)
            logger.debug(
                f"APPLY [Mask -1]: Block {block.id} | Location: {location} | "
                f"Field {field_id.name} set to {float(val):.4e}"
            )


def _detect_boundary_location(block: StencilBlock) -> str:
    """Identifies the spatial boundary face based on adjacent ghost topology."""
    if getattr(block.i_minus, "is_ghost", False): return "x_min"
    if getattr(block.i_plus, "is_ghost", False):  return "x_max"
    if getattr(block.j_minus, "is_ghost", False): return "y_min"
    if getattr(block.j_plus, "is_ghost", False):  return "y_max"
    if getattr(block.k_minus, "is_ghost", False): return "z_min"
    if getattr(block.k_plus, "is_ghost", False):  return "z_max"
    return "wall"


def _resolve_boundary_values(boundary_cfg: list, domain_cfg: dict, location: str) -> dict:
    """Matches face location against sovereign JSON configurations."""
    domain_type = domain_cfg.get("type") if domain_cfg else None

    # Handle EXTERNAL domain free-stream conditions
    if domain_type == "EXTERNAL" and location != "wall":
        ref_v = domain_cfg.get("reference_velocity", [0.0, 0.0, 0.0])
        return {"u": ref_v[0], "v": ref_v[1], "w": ref_v[2]}

    # Handle standard boundary configuration matching
    if isinstance(boundary_cfg, list):
        for bc in boundary_cfg:
            if bc.get("location") == location:
                return bc.get("values", {})

    return {}
