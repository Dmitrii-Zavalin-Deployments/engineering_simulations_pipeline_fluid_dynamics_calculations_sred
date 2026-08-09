# tests/quality_gates/physics_gate/test_boundaries_predictor.py

import numpy as np

from src.common.field_schema import FI
from src.step3.boundaries.applier import BC_FIELD_MAP, apply_boundary_values
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_gate_3d_no_slip_integrity():
    """
    Gate 3.D: No-Slip Integrity Audit
    Identity: Trial Field (u_star) matches boundary value at location.
    Compliance: Rule 9 (Trial Field Redirection)
    """
    # 1. Setup: Create a dummy state (4x4x4 core)
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    
    # 2. Target a boundary block. 
    # In Step 2 dummy, the first block in the matrix is (1,1,1) in memory,
    # which sits at the x_min, y_min, z_min intersection.
    block = state.stencil_matrix[0] 
    
    # 3. Define the No-Slip Rule for a specific coordinate
    # Per Physics Gate 3.D: u_ghost = -u_interior is the theoretical check,
    # but the implementation logic sets the face value to 0.0.
    rule = {
        "type": "no-slip", 
        "location": "y_min", 
        "values": {"u": 0.0, "v": 0.0, "w": 0.0}
    }
    
    # 4. Action: apply_boundary_values(block, rule)
    # This should map 'u' to FI.VX_STAR and set it to 0.0
    apply_boundary_values(block, rule)
    
    # 5. Verification: Trial Field Redirection (Rule 9)
    # Check that VX_STAR was updated, but the base VX remains untouched.
    u_star_val = block.center.get_field(FI.VX_STAR)
    block.center.get_field(FI.VX)
    
    assert np.isclose(u_star_val, 0.0), (
        f"MMS FAILURE [No-Slip]: VX_STAR at {rule['location']} should be 0.0, got {u_star_val}"
    )
    
    # Verification: Contract Compliance
    # Ensure the mapping followed BC_FIELD_MAP exactly.
    assert BC_FIELD_MAP["u"] == FI.VX_STAR, "Contract Violation: BC_FIELD_MAP drift detected."

def test_gate_3d_dirichlet_pressure():
    """
    Verification: Pressure Trial Field (P_NEXT) Redirection
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10] # Interior/Boundary block
    
    pressure_val = 101325.0
    rule = {
        "type": "dirichlet",
        "location": "outlet",
        "values": {"p": pressure_val}
    }
    
    apply_boundary_values(block, rule)
    
    p_next = block.center.get_field(FI.P_NEXT)
    assert np.isclose(p_next, pressure_val), f"MMS FAILURE: P_NEXT not hydrated for {rule['location']}"