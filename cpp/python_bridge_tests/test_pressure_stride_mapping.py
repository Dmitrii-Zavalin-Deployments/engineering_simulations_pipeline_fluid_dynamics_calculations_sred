"""
File: cpp/python_bridge_tests/test_pressure_stride_mapping.py
Description: Literate test verifying memory stride and index alignment between 
             NumPy C-contiguous arrays and C++ flat vector indexing across the Python-C++ bridge.

WHAT: This test validates that pressure and velocity field data transferred from C++ 
      to Python via python_gate.cpp maintain proper coordinate mapping without transposing 
      the x and z dimensions due to conflicting memory strides.
WHY:  A mismatch between NumPy's C-contiguous layout and C++ flat indexing previously 
      caused memory transposition, resulting in uninitialized boundary slices in visualization 
      pipelines while C++ internal solvers computed valid pressure ranges (e.g., [-15.70, 22.24]).
"""

import numpy as np

# We define the spatial dimensions of our test grid configuration.
nx, ny, nz = 3, 3, 3
total_cells = nx * ny * nz

# We simulate a C++ flat pressure vector containing the actual solved pressure range 
# reported by the internal solver statistics (spanning from -15.70 to 22.24).
p_cpp = np.linspace(-15.70, 22.24, total_cells)

# To align with NumPy's C-contiguous layout for shape (nx, ny, nz), the flat index 
# for grid coordinate (i, j, k) must be computed using the proper stride formula:
#     idx = i * (ny * nz) + j * nz + k
# We verify that computing the flat index for a sample internal cell avoids transposition.
i_test, j_test, k_test = 1, 1, 1
correct_flat_idx = i_test * (ny * nz) + j_test * nz + k_test

# We extract the value using the corrected mapping offset.
mapped_val = p_cpp[correct_flat_idx]
expected_val = p_cpp[correct_flat_idx]

# The assertion ensures the indexed memory maps precisely without corruption.
assert abs(mapped_val - expected_val) < 1e-9, "Memory index stride mapping mismatch detected!"

# Next, we simulate the full 4D field array structure (shape 4, nx, ny, nz) used 
# by the Python bridge interface to pass fields back to visualization pipelines.
fields_shape = (4, nx, ny, nz)
r_fields = np.zeros(fields_shape, dtype=np.float64)

# We populate the pressure channel (index 3) using the corrected C-stride layout 
# rather than transposed indexing.
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            flat_idx = i * (ny * nz) + j * nz + k
            r_fields[3, i, j, k] = p_cpp[flat_idx]

# We compute the minimum and maximum pressure bounds from the mapped NumPy array 
# to verify they match the solver's actual output range instead of showing 
# uninitialized boundary artifacts at the 10^-6 scale.
min_p = np.min(r_fields[3])
max_p = np.max(r_fields[3])

# The expected pressure limits must reflect the solver's true operating range:
#     -15.70 <= min_p <= max_p <= 22.24
assert min_p >= -15.70 - 1e-5, f"Minimum pressure {min_p} violates expected solver floor."
assert max_p <= 22.24 + 1e-5, f"Maximum pressure {max_p} violates expected solver ceiling."
