"""
@file test_bindings.py
@brief Literate Test Suite for Python Pybind11 Bindings Bridge

This test file acts as a narrative document. Explanatory text and physical
principles are written as commented prose, while the executable Python assertions
verify correct interaction between the Python runtime and the C++ predictor kernel.
"""

import numpy as np
import pytest

try:
    import navier_stokes_cpp
except ImportError:
    navier_stokes_cpp = None

# ============================================================================
# NARRATIVE SECTION 1: Extension Module Availability and Initialization
# ============================================================================
# The pybind11 module must successfully load into the Python interpreter,
# registering the high-performance C++ functions and module documentation string.
# ============================================================================

def test_module_initialization():
    assert navier_stokes_cpp is not None, "Extension module navier_stokes_cpp must be compiled and available."
    assert isinstance(navier_stokes_cpp.__doc__, str)
    assert len(navier_stokes_cpp.__doc__) > 0

# ============================================================================
# NARRATIVE SECTION 2: Predictor Kernel Python Bridge Execution
# ============================================================================
# The Python bridge function computes trial velocities by mapping 4D NumPy arrays 
# to raw C++ pointers and initializing GridDimensions and FluidProperties 
# where kinematic viscosity is ν = μ / ρ.
# ============================================================================

def test_compute_predictor_kernel_cpp_execution():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 5, 5, 5
    dx, dy, dz = 0.1, 0.1, 0.1
    dt = 0.01
    rho = 1.03
    mu = 0.001

    fields = np.zeros((3, nx, ny, nz), dtype=np.float64)
    fields[0, :, :, :] = 1.0
    fields[1, :, :, :] = 0.5
    fields[2, :, :, :] = 0.2

    mask = np.zeros((nx, ny, nz), dtype=np.int32)

    navier_stokes_cpp.compute_predictor_kernel_cpp(
        fields, mask,
        nx, ny, nz,
        dx, dy, dz,
        dt, rho, mu
    )

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                assert abs(fields[0, i, j, k] - 1.0) < 1e-12
                assert abs(fields[1, i, j, k] - 0.5) < 1e-12
                assert abs(fields[2, i, j, k] - 0.2) < 1e-12
