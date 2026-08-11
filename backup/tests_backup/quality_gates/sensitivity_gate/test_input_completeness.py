# tests/quality_gates/sensitivity_gate/test_input_completeness.py

import pytest

from src.step1.helpers import generate_3d_masks
from tests.helpers.solver_input_schema_dummy import create_validated_input


def test_gate_1a_mask_size_mandate():
    """
    Gate 1.A: Domain Audit (Mask Size Mandate)
    
    Verification: Catch ValueError in src/step1/helpers.py when volume != nx*ny*nz.
    Compliance: Rule 7 (Atomic Truth) - Topology Protection.
    Compliance: Rule 5 (No Logical Defaults) - Explicit Volume check.
    """

    # 1. Setup a controlled 3x3x3 grid (Core Volume = 27)
    nx, ny, nz = 3, 3, 3
    context = create_validated_input()
    
    # Pathing Fix: Manual hydration of the SSoT Grid object
    context.grid.nx = nx
    context.grid.ny = ny
    context.grid.nz = nz
    
    grid = context.grid
    
    # 2. Create "Bad Data" (20 cells instead of 27)
    # This simulates a malformed raw JSON intake (e.g., truncated data).
    bad_mask_data = [1] * 20 
    
    # 3. Verification: Ensure the Firewall triggers the exact SSoT error message.
    # The helper must identify that 20 != 3*3*3 before any memory allocation.
    expected_error = "Mask data size mismatch: Expected 27 cells, got 20"
    
    with pytest.raises(ValueError, match=expected_error):
        generate_3d_masks(bad_mask_data, grid)

def test_gate_1a_perfect_match_pass():
    """
    Verification: Ensure valid mask data passes the size mandate without error.
    Compliance: Rule 4 (SSoT Hierarchy) - Consistent shape mapping.
    """

    # 1. Setup: Define a 2x2x2 core (Volume = 8)
    nx, ny, nz = 2, 2, 2
    context = create_validated_input()
    
    context.grid.nx = nx
    context.grid.ny = ny
    context.grid.nz = nz
    
    grid = context.grid
    
    # Define valid input: 8 cells exactly.
    valid_mask_data = [1, 1, 0, 0, -1, -1, 1, 1] 
    
    # 2. Action: This should execute without raising a ValueError.
    # It must return the 3D reshaped mask and the derived boolean arrays.
    mask_3d, is_fluid, is_boundary = generate_3d_masks(valid_mask_data, grid)
    
    # 3. Verification: Structural Parity
    # Verify the reshape logic aligns with the core dimensions (nz, ny, nx).
    assert mask_3d.shape == (nz, ny, nx), (
        f"MMS FAILURE: Mask shape {mask_3d.shape} does not match (nz, ny, nx)."
    )
    assert mask_3d.size == 8, "MMS FAILURE: Mask size drift detected."
    
    # Ensure boolean arrays were hydrated correctly (Logic Layer readiness)
    assert is_fluid.dtype == bool
    assert is_boundary.dtype == bool

def test_gate_1a_empty_input_firewall():
    """
    Verification: Ensure the firewall catches empty or null input arrays.
    Compliance: Rule 5 (Zero-Default Policy).
    """
    nx, ny, nz = 2, 2, 2
    context = create_validated_input()
    
    context.grid.nx = nx
    context.grid.ny = ny
    context.grid.nz = nz
    
    grid = context.grid
    
    # The volume check (2*2*2=8) must fail against an empty list.
    with pytest.raises(ValueError, match="Expected 8 cells, got 0"):
        generate_3d_masks([], grid)