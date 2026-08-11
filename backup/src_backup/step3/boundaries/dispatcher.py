# src/step3/boundaries/dispatcher.py

import logging

from src.common.stencil_block import StencilBlock

logger = logging.getLogger(__name__)


def get_applicable_boundary_configs(block: StencilBlock, boundary_cfg: list, grid, domain_cfg: dict) -> list:
    domain_type = None
    if domain_cfg:
        domain_type = domain_cfg.get("type")

    # Spatial face detection (ghost-based)
    b_type = _get_domain_location_type(block, grid)

    # --- EXTERNAL DOMAINS: Free-stream + spatial-first ---
    if domain_type == "EXTERNAL":
        if b_type != "none":
            try:
                ref_v = domain_cfg["reference_velocity"]
            except KeyError:
                # Missing required reference velocity for EXTERNAL domain
                raise KeyError("Missing 'reference_velocity' for EXTERNAL domain") from None

            return [{
                "location": b_type,
                "type": "free-stream",
                "values": {"u": ref_v[0], "v": ref_v[1], "w": ref_v[2]},
            }]

        # No ghost neighbors in EXTERNAL: treat as interior fluid
        return [{
            "location": "interior",
            "type": "fluid_gas",
            "values": {},
        }]

    # --- INTERNAL / UNSPECIFIED DOMAINS ---
    # Rule: Spatial boundary must take priority over mask when a ghost neighbor exists.
    if b_type != "none":
        try:
            return _find_config(boundary_cfg, b_type)
        except KeyError:
            # Strict config requirement for spatial faces
            raise KeyError(f"Missing boundary definition for {b_type}") from None

    # No spatial boundary: apply mask-based physical axioms
    mask = block.center.mask

    if mask == -1:
        logger.debug(f"DISPATCH [Mask]: Block {block.id} treated as Wall (mask -1)")
        return _find_config(boundary_cfg, "wall")

    if mask == 0:
        logger.debug(f"DISPATCH [Mask]: Block {block.id} treated as Solid (mask 0)")
        return [{
            "location": "solid",
            "type": "no-slip",
            "values": {"u": 0.0, "v": 0.0, "w": 0.0},
        }]

    # Default: interior fluid
    return [{
        "location": "interior",
        "type": "fluid_gas",
        "values": {},
    }]


def _find_config(boundary_cfg: list, location: str) -> list:
    """
    Direct lookup: Matches location string exactly as defined in the Schema.
    No normalization. Rule 5: If it's not in the JSON, it's an error.
    """
    for bc in boundary_cfg:
        if bc["location"] == location:
            return [bc]

    raise KeyError(f"No boundary configuration found for location: '{location}'")


def _get_domain_location_type(block: StencilBlock, grid) -> str:
    """
    Identifies the boundary face based strictly on Neighbor-State detection.
    Rule 4: Topology defines Logic. Coordinate math is rejected to prevent drift.
    """
    # Primary & Only Authority: Ghost Neighbor Detection
    if block.i_minus.is_ghost:
        return "x_min"
    if block.i_plus.is_ghost:
        return "x_max"
    if block.j_minus.is_ghost:
        return "y_min"
    if block.j_plus.is_ghost:
        return "y_max"
    if block.k_minus.is_ghost:
        return "z_min"
    if block.k_plus.is_ghost:
        return "z_max"

    # If no neighbors are ghosts, this is either an interior cell
    # or an internal masked obstruction (handled by mask logic in caller).
    return "none"
