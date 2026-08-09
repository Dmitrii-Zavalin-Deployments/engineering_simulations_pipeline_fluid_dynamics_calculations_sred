# tests/step1/test_helpers.py

from unittest.mock import patch

import numpy as np
import pytest

from src.common.solver_input import GridInput
from src.step1.helpers import generate_3d_masks


def create_test_grid(nx, ny, nz):
    """
    Hydrates a GridInput object following the ValidatedContainer pattern.
    Rule 5: Deterministic Initialization.
    """
    grid = GridInput()
    
    # We must set all values to satisfy the ValidatedContainer 
    # before the object is used in the helper functions.
    grid.nx = int(nx)
    grid.ny = int(ny)
    grid.nz = int(nz)
    
    # Define physical bounds
    grid.x_min, grid.x_max = 0.0, float(nx)
    grid.y_min, grid.y_max = 0.0, float(ny)
    grid.z_min, grid.z_max = 0.0, float(nz)
    
    return grid

def test_generate_3d_masks_basic():
    """Verify that a simple 2x2x2 mask maps correctly via SSoT get_coords."""
    grid = create_test_grid(2, 2, 2)
    # Flat: [1, 0, -1, 1, 0, -1, 1, 0] (Length 8 for 2x2x2)
    mask_data = [1, 0, -1, 1, 0, -1, 1, 0]
    mask_3d, is_fluid, is_boundary = generate_3d_masks(mask_data, grid)
    
    # Testing coordinate mapping from get_coords_from_index:
    # Index 0 -> (0,0,0) -> 1
    assert mask_3d[0, 0, 0] == 1
    # Index 2 -> (0,1,0) -> -1
    assert mask_3d[0, 1, 0] == -1
    # Index 5 -> (1,0,1) -> -1
    assert mask_3d[1, 0, 1] == -1
    
    # np.sum(is_fluid) counts values where mask == 1
    # Indices 0, 3, 6 are '1' in mask_data
    assert np.sum(is_fluid) == 3
    assert np.sum(is_boundary) == 2

def test_generate_3d_masks_empty():
    """Verify behavior with an empty mask list triggers size integrity check."""
    grid = create_test_grid(1, 1, 1)
    with pytest.raises(ValueError, match="Mask data size mismatch"):
        generate_3d_masks([], grid)

def test_generate_3d_masks_overflow():
    """Verify that providing too much data raises a ValueError (Rule 5)."""
    grid = create_test_grid(1, 1, 1)
    # Only 1 cell expected (1*1*1), providing 2
    with pytest.raises(ValueError, match="Mask data size mismatch"):
        generate_3d_masks([1, 0], grid)

def test_mapping_integrity_sequence():
    """Verify sequential mapping via SSoT logic (i, j, k)."""
    nx, ny, nz = 3, 2, 2
    grid = create_test_grid(nx, ny, nz)
    
    # Create data where value == index to track movement
    mask_data = list(range(nx * ny * nz))
    mask_3d, _, _ = generate_3d_masks(mask_data, grid)
    
    # Check start and end boundaries
    assert mask_3d[0, 0, 0] == 0
    # Final flat index is (nx*ny*nz)-1
    assert mask_3d[nx-1, ny-1, nz-1] == (nx * ny * nz) - 1

def test_different_dimensions():
    """Test non-cubic grid dimensions (4x2x1)."""
    grid = create_test_grid(4, 2, 1)
    # Value is (Value + 1) to distinguish from zero-init
    mask_data = [1, 2, 3, 4, 5, 6, 7, 8]
    mask_3d, _, _ = generate_3d_masks(mask_data, grid)
    
    # (i, j, k) mapping for index 7 in 4x2x1:
    # k = 7 // (4*2) = 0
    # rem = 7 % 8 = 7
    # j = 7 // 4 = 1
    # i = 7 % 4 = 3
    assert mask_3d[3, 1, 0] == 8

def test_generate_3d_masks_debug_path():
    """
    Targets lines 47-49 in src/step1/helpers.py.
    Uses patch to force the DEBUG global to True.
    """
    grid = create_test_grid(2, 2, 2)
    mask_data = [1] * 8
    
    # We patch 'DEBUG' in the namespace where it is USED
    with patch("src.step1.helpers.DEBUG", True):
        mask_3d, is_fluid, is_boundary = generate_3d_masks(mask_data, grid)
        
        # Verify result integrity while DEBUG is active
        assert mask_3d.shape == (2, 2, 2)
        assert np.all(is_fluid)
        assert not np.any(is_boundary)

def test_generate_3d_masks_mapping_overflow_protection():
    """
    Targets line 40 in src/step1/helpers.py.
    Tests the logic guard against index drift/SSoT mapping errors.
    """
    grid = create_test_grid(2, 2, 2)
    mask_data = [1] * 8
    
    # We mock get_coords_from_index to return coordinates OUTSIDE the grid
    # This simulates a failure in the coordinate mapping logic.
    with patch("src.step1.helpers.get_coords_from_index") as mock_mapping:
        # Force an out-of-bounds coordinate (e.g., i=5 on a 2x2x2 grid)
        mock_mapping.return_value = (5, 0, 0)
        
        with pytest.raises(ValueError, match="Mask mapping overflow"):
            generate_3d_masks(mask_data, grid)