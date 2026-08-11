# tests/common/test_grid_math.py

import numpy as np
import pytest

from src.common.grid_math import get_coords_from_index, get_flat_index

# --- SSoT Constants for Testing ---
# Base configuration (Interior)
NX, NY, NZ = 4, 4, 4
# Buffered configuration (Halo/Ghost cells: +2 to each dimension)
# This mimics the "Foundation" memory layout
BUF_NX, BUF_NY = NX + 2, NY + 2
BUF_NZ = NZ + 2

def test_manual_mapping_verification():
    """
    Verify specific known values for 3D to 1D and vice-versa.
    Rule 5: Explicit verification of row-major indexing: i + nx*j + (nx*ny)*k
    """
    # Test Point (i, j, k) = (2, 2, 2) in a 4x4 grid (NX=4, NY=4)
    # Calculation: 2 + (4 * 2) + (16 * 2) = 2 + 8 + 32 = 42
    idx = get_flat_index(2, 2, 2, NX, NY)
    assert idx == 42
    
    coords = get_coords_from_index(42, NX, NY)
    assert coords == (2, 2, 2)

def test_volume_boundaries():
    """Verify that origin and terminal points map without overflow/underflow."""
    # Origin (0,0,0) -> 0
    assert get_flat_index(0, 0, 0, NX, NY) == 0
    assert get_coords_from_index(0, NX, NY) == (0, 0, 0)
    
    # Last element (3,3,3) in a 4x4x4 volume
    max_idx = (NX * NY * NZ) - 1
    assert get_flat_index(NX-1, NY-1, NZ-1, NX, NY) == max_idx
    assert get_coords_from_index(max_idx, NX, NY) == (NX-1, NY-1, NZ-1)

def test_halo_offset_integrity():
    """
    CRITICAL: Verify the +1 offset logic used for Ghost/Halo cells.
    Ensures that interior index (0,0,0) maps to buffered index (1,1,1) 
    correctly within a 6x6x6 buffered volume.
    """
    # In a buffered grid (NX=6, NY=6), (1,1,1) is:
    # 1 + (6 * 1) + (36 * 1) = 43
    buf_idx = get_flat_index(1, 1, 1, BUF_NX, BUF_NY)
    assert buf_idx == 43
    
    coords = get_coords_from_index(43, BUF_NX, BUF_NY)
    assert coords == (1, 1, 1), f"Ghost cell coordinate drift: expected (1,1,1), got {coords}"

@pytest.mark.parametrize("i, j, k", [
    (0, 0, 0), (3, 3, 3), (1, 2, 3), (0, 3, 2), (2, 0, 1)
])
def test_round_trip_coordination(i, j, k):
    """Verify that 3D coordinates survive the transition to 1D and back."""
    flat = get_flat_index(i, j, k, NX, NY)
    coords = get_coords_from_index(flat, NX, NY)
    assert coords == (i, j, k)

def test_buffered_foundation_simulation():
    """
    Simulates placing data in a buffered foundation and retrieving it.
    This validates the bridge between grid_math and actual memory arrays.
    """
    total_buf_cells = BUF_NX * BUF_NY * BUF_NZ
    foundation = np.zeros(total_buf_cells)
    
    # Map interior points to the buffered foundation (i+1, j+1, k+1)
    test_points = [(0, 0, 0), (2, 1, 3), (NX-1, NY-1, NZ-1)]
    marker_value = 144.93  # Distinctive marker
    
    for (i, j, k) in test_points:
        # Step into the buffered region
        flat_idx = get_flat_index(i + 1, j + 1, k + 1, BUF_NX, BUF_NY)
        foundation[flat_idx] = marker_value
        
        # Verify index-to-coord roundtrip
        retrieved_coords = get_coords_from_index(flat_idx, BUF_NX, BUF_NY)
        assert retrieved_coords == (i + 1, j + 1, k + 1)
        
        # Verify data retrieval
        assert foundation[flat_idx] == marker_value

def test_exhaustive_volume_scan():
    """Exhaustive check: ensures no collisions or gaps in a full 4x4x4 volume."""
    indices = set()
    total_cells = NX * NY * NZ
    
    for idx in range(total_cells):
        i, j, k = get_coords_from_index(idx, NX, NY)
        flat_back = get_flat_index(i, j, k, NX, NY)
        
        assert flat_back == idx, f"Collision/Misalignment at index {idx}"
        indices.add(flat_back)
    
    assert len(indices) == total_cells, "Incomplete volume coverage."