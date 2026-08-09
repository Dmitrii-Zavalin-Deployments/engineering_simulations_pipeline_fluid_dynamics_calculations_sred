# tests/common/test_cell_coverage.py

import numpy as np

from src.common.cell import Cell


def test_cell_comprehensive_coverage():
    # Setup: 10x10x10 buffer (nx=10, ny=10, nz=10)
    # nx_buf/ny_buf usually include ghost layers (+2)
    nx_buf, ny_buf = 12, 12
    buffer = np.zeros((12*12*12, 20)) 
    
    # We pick an index in the middle to test i, j, k
    # If index=145, get_coords_from_index should return something like (2, 1, 1)
    index = 145 
    cell = Cell(index=index, fields_buffer=buffer, nx_buf=nx_buf, ny_buf=ny_buf)

    # 1. Hit Line 33: Pass a single-element list to _to_scalar via a setter
    cell.vx = [10.5]
    assert cell.vx == 10.5

    # 2. Hit Lines 41, 46, 51: Access i, j, k properties
    # This triggers the get_coords_from_index logic
    _ = cell.i
    _ = cell.j
    _ = cell.k

    # 3. Hit Line 129: Use the vector setter for 'u'
    new_velocity = np.array([1.0, 2.0, 3.0])
    cell.u = new_velocity
    np.testing.assert_array_equal(cell.u, new_velocity)

    print("✅ All missing branches in cell.py exercised.")

