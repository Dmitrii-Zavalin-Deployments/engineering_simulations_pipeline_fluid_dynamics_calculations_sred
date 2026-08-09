# tests/quality_gates/sensitivity_gate/test_mask_integrity.py

import numpy as np
import pytest

from src.step1.helpers import generate_3d_masks
from tests.helpers.solver_input_schema_dummy import create_validated_input


def test_gate_1b_overflow_guard_logic():
    """
    Gate 1.B: Topology Audit (Overflow Guard)
    
    Verification: Catch 'Mask mapping overflow' in src/step1/helpers.py.
    Compliance: Rule 7 (Atomic Truth) - Topology Protection.
    Compliance: Rule 4 (SSoT Hierarchy) - Mapping Consistency.
    """

    # 1. Setup a valid 2x2x2 grid structure (Rule 5: Explicit Init)
    nx, ny, nz = 2, 2, 2
    context = create_validated_input()
    
    # Pathing Fix: Ensure we traverse through .input_data to satisfy Rule 4
    context.grid.nx = nx
    context.grid.ny = ny
    context.grid.nz = nz
    
    grid = context.grid
    
    # Valid input: 8 cells (2*2*2)
    valid_mask = [1] * 8
    
    # 2. Action: Test successful 1D-to-3D mapping integrity
    mask_3d, is_fluid, _ = generate_3d_masks(valid_mask, grid)
    
    # 3. Verification: Structural Parity
    # Reshape must follow (nz, ny, nx) to align with stride-based logic (Rule 7)
    assert mask_3d.shape == (nz, ny, nx), (
        f"Topology Breach: 3D Shape {mask_3d.shape} mismatch with core ({nz}, {ny}, {nx})."
    )
    assert np.all(is_fluid), "Topology Breach: Fluid mask vectorization failed (Rule 1)."

def test_gate_1b_mapping_overflow_trigger(monkeypatch):
    """
    Verification: Force a mapping overflow to ensure the deterministic 
    validation in Step 1 catches index drift.
    Compliance: Physical Logic Firewall - Out-of-bounds protection.
    """

    # 1. Setup: Explicit dimensions via SSoT path
    nx, ny, nz = 2, 2, 2
    context = create_validated_input()
    context.grid.nx = nx
    context.grid.ny = ny
    context.grid.nz = nz
    
    grid = context.grid
    valid_mask = [1] * 8
    
    # 2. Action: Monkeypatch the coordinate getter to return an out-of-bounds index (i=5)
    # This simulates a 'Mapping Overflow' scenario where a logic error 
    # attempts to write outside the (2,2,2) allocated core.
    def mock_coords(idx, current_nx, current_ny):
        return (5, 0, 0) # i=5 is > nx=2
    
    # Ensure the path to the helper is accurate to your project structure
    monkeypatch.setattr("src.step1.helpers.get_coords_from_index", mock_coords)
    
    # 3. Verification: The Firewall must raise a ValueError before the reshape fails.
    # Success Metric: Exact identification of the offending index and coordinate.
    expected_msg = r"Mask mapping overflow at index 0 -> \(5, 0, 0\)"
    with pytest.raises(ValueError, match=expected_msg):
        generate_3d_masks(valid_mask, grid)

def test_gate_1b_padding_integrity():
    """
    Verification: Ensure that topological padding (Ghost Data guard) 
    aligns with expected buffer bounds.
    Compliance: Rule 9 (Hybrid Memory Foundation - Ghost padding parity).
    """

    # 1. Setup: 2x2x2 core
    nx, ny, nz = 2, 2, 2
    context = create_validated_input()
    context.grid.nx = nx
    context.grid.ny = ny
    context.grid.nz = nz
    
    grid = context.grid
    valid_mask = [1] * 8
    
    # 2. Action: Hydrate the 3D core
    mask_3d, _, _ = generate_3d_masks(valid_mask, grid)
    
    # Apply a standard 1-cell pad for ghost boundaries (Rule 9)
    # This simulates the internal padding logic used by orchestrate_step1.
    padded = np.pad(mask_3d, pad_width=1, mode="constant", constant_values=-1)
    
    # 3. Verification: Ghost-Cell Lock
    # Core (2,2,2) with pad 1 becomes (4,4,4) total volume.
    expected_padded_shape = (nz + 2, ny + 2, nx + 2)
    assert padded.shape == expected_padded_shape, (
        f"Topology Breach: Expected {expected_padded_shape}, got {padded.shape}"
    )
    
    # Verify the padding values (the "Ghost" layer) are set correctly at the boundary.
    # index [0,0,0] is a ghost cell.
    assert padded[0, 0, 0] == -1, "Ghost Data Guard: Padding value mismatch at boundary."
    # index [1,1,1] is the first physical cell.
    assert padded[1, 1, 1] == 1, "Ghost Data Guard: Core value corrupted by padding."