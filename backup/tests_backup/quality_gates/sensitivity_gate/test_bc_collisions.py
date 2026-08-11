import logging

import pytest

from src.step3.boundaries.dispatcher import get_applicable_boundary_configs
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

# Tracking the specific dispatcher logger
DISPATCH_LOGGER = "src.step3.boundaries.dispatcher"


def _force_interior(block):
    """
    Override Step-2 dummy ghost flags so the block behaves like a true interior block.
    The dummy builder marks many blocks as boundary-adjacent (ghost neighbors),
    but these tests require interior behavior.
    """
    block.i_minus.is_ghost = False
    block.i_plus.is_ghost = False
    block.j_minus.is_ghost = False
    block.j_plus.is_ghost = False
    block.k_minus.is_ghost = False
    block.k_plus.is_ghost = False
    return block


def test_gate_3a_3b_dispatcher_mask_symmetry(caplog):
    """
    Gate 3.A & 3.B: Dispatcher-Mask Symmetry Audit.
    Verification: Ensure mask values (-1, 0) trigger the correct Axiomatic logic.
    """
    caplog.set_level(logging.DEBUG, logger=DISPATCH_LOGGER)
    state = make_step2_output_dummy(nx=10, ny=10, nz=10)

    boundary_cfg = [
        {"location": "wall", "type": "no-slip", "values": {"u": 0.1}}
    ]
    domain_cfg = {"type": "INTERNAL"}

    # Force block 500 to behave as interior
    block = _force_interior(state.stencil_matrix[500])

    # --- Step 3.A: Wall logic ---
    block.center.mask = -1
    rules = get_applicable_boundary_configs(block, boundary_cfg, state.grid, domain_cfg)

    assert "DISPATCH [Mask]" in caplog.text
    assert "treated as Wall (mask -1)" in caplog.text
    assert rules[0]["values"]["u"] == 0.1

    caplog.clear()

    # --- Step 3.B: Solid logic ---
    block.center.mask = 0
    rules = get_applicable_boundary_configs(block, boundary_cfg, state.grid, domain_cfg)

    assert "DISPATCH [Mask]" in caplog.text
    assert "treated as Solid (mask 0)" in caplog.text
    assert rules[0]["location"] == "solid"
    assert rules[0]["values"]["u"] == 0.0


def test_gate_3c_interior_fluid_axiom():
    """
    Gate 3.C: Interior Fluid Axiom Audit.
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)

    # Force block 10 to behave as interior
    block = _force_interior(state.stencil_matrix[10])
    block.center.mask = 1

    rules = get_applicable_boundary_configs(block, [], state.grid, {"type": "INTERNAL"})

    assert rules[0]["location"] == "interior"
    assert rules[0]["type"] == "fluid_gas"
    assert rules[0]["values"] == {}


def test_gate_3d_external_flow_axiom():
    """
    Gate 3.D: External Flow (Far-field) Axiom Audit.
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)

    # Block 0 is intentionally a boundary block for EXTERNAL tests
    block = state.stencil_matrix[0]

    domain_cfg = {
        "type": "EXTERNAL",
        "reference_velocity": [10.0, 0.0, 0.0],
    }

    rules = get_applicable_boundary_configs(block, [], state.grid, domain_cfg)

    assert rules[0]["type"] == "free-stream"
    assert rules[0]["values"]["u"] == 10.0


def test_gate_3a_missing_wall_config_collision():
    """
    Step 3.A: Catch KeyError when mask -1 is present but 'wall' config is missing.
    """
    state = make_step2_output_dummy(nx=10, ny=10, nz=10)

    # Force block 500 to behave as interior
    block = _force_interior(state.stencil_matrix[500])
    block.center.mask = -1

    incomplete_cfg = [
        {"location": "x_min", "type": "no-slip", "values": {"u": 0}},
        {"location": "x_max", "type": "no-slip", "values": {"u": 0}},
        {"location": "y_min", "type": "no-slip", "values": {"v": 0}},
        {"location": "y_max", "type": "no-slip", "values": {"v": 0}},
        {"location": "z_min", "type": "no-slip", "values": {"w": 0}},
        {"location": "z_max", "type": "no-slip", "values": {"w": 0}},
    ]

    expected_error = "No boundary configuration found for location: 'wall'"

    with pytest.raises(KeyError, match=expected_error):
        get_applicable_boundary_configs(
            block,
            incomplete_cfg,
            state.grid,
            {"type": "INTERNAL"},
        )
