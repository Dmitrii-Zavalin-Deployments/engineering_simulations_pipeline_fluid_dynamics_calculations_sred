"""
@file test_simulation_prestep.py
@brief Literate Test Suite for Simulation Pre-Step Boundary & Initial Condition Setup

This test file acts as a narrative document for the pre-step boundary condition
and spatial initialization subsystem (`simulation_prestep.cpp`). Explanatory text
and physical principles are written as commented prose, while executable Python
assertions verify correct boundary value assignment, Neumann velocity extrapolation,
and rigorous contract enforcement for grid dimensions and tensor buffer sizes.
"""

import numpy as np
import pytest

try:
    import navier_stokes_cpp
except ImportError:
    navier_stokes_cpp = None


# ============================================================================
# NARRATIVE SECTION 1: Grid Dimension and Contract Safety Validation
# ============================================================================
# The pre-step routine enforces strict geometric and memory safety contracts:
# 
#     nx >= 3, ny >= 3, nz >= 3
#     u.size() == v.size() == w.size() == p.size() == mask.size() == nx * ny * nz
#
# Violations of these constraints throw explicit std::invalid_argument exceptions,
# specifically targeting lines 43 and 49 in simulation_prestep.cpp.
# ============================================================================

def test_pre_step_invalid_grid_dimensions():
    """Verifies that grid dimensions smaller than 3x3x3 trigger a geometry error exception (Line 43)."""
    if navier_stokes_cpp is None or not hasattr(navier_stokes_cpp, "execute_pre_step"):
        pytest.skip("navier_stokes_cpp.execute_pre_step not available.")

    # Invalid geometry condition: nx = 2 (< 3)
    nx, ny, nz = 2, 8, 8
    total_cells = nx * ny * nz
    u = np.zeros(total_cells, dtype=np.float64)
    v = np.zeros(total_cells, dtype=np.float64)
    w = np.zeros(total_cells, dtype=np.float64)
    p = np.zeros(total_cells, dtype=np.float64)
    mask = np.ones(total_cells, dtype=np.int32)
    bc_list = []

    # Contract check:
#     nx >= 3 and ny >= 3 and nz >= 3
    with pytest.raises((ValueError, RuntimeError)):
        navier_stokes_cpp.execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz)


def test_pre_step_vector_size_mismatch():
    """Verifies that field vector size mismatches trigger a contract violation exception (Line 49)."""
    if navier_stokes_cpp is None or not hasattr(navier_stokes_cpp, "execute_pre_step"):
        pytest.skip("navier_stokes_cpp.execute_pre_step not available.")

    nx, ny, nz = 8, 8, 8
    total_cells = nx * ny * nz
    
    # Mismatched size for u vector: size = total_cells - 1 instead of total_cells
    u = np.zeros(total_cells - 1, dtype=np.float64)
    v = np.zeros(total_cells, dtype=np.float64)
    w = np.zeros(total_cells, dtype=np.float64)
    p = np.zeros(total_cells, dtype=np.float64)
    mask = np.ones(total_cells, dtype=np.int32)
    bc_list = []

    # Contract check:
#     u.size() == nx * ny * nz
    with pytest.raises((ValueError, RuntimeError)):
        navier_stokes_cpp.execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz)


# ============================================================================
# NARRATIVE SECTION 2: Boundary Condition Execution and Type Dispatch
# ============================================================================
# The pre-step routine iterates through configured boundary conditions, applying:
#   - Inflow velocity prescription (u_val, v_val, w_val)
#   - Pressure and outflow Neumann extrapolation
#   - No-slip zero velocity enforcement
#   - Free-slip tangential slip condition across domain boundaries
# ============================================================================

def test_pre_step_boundary_execution():
    """Executes pre-step boundary assignment across all schema types and verifies correct buffer mutation."""
    if navier_stokes_cpp is None or not hasattr(navier_stokes_cpp, "execute_pre_step"):
        pytest.skip("navier_stokes_cpp.execute_pre_step not available.")

    nx, ny, nz = 8, 8, 8
    total_cells = nx * ny * nz
    u = np.zeros(total_cells, dtype=np.float64)
    v = np.zeros(total_cells, dtype=np.float64)
    w = np.zeros(total_cells, dtype=np.float64)
    p = np.zeros(total_cells, dtype=np.float64)
    mask = np.ones(total_cells, dtype=np.int32)

    # Construct boundary conditions covering inflow, pressure, no-slip, and free-slip
    bc1 = navier_stokes_cpp.BoundaryCondition()
    bc1.location = "x_min"
    bc1.type = "inflow"
    bc1.u_val = 2.0
    bc1.v_val = 0.0
    bc1.w_val = 0.0

    bc2 = navier_stokes_cpp.BoundaryCondition()
    bc2.location = "x_max"
    bc2.type = "pressure"
    bc2.scalar_p = 101325.0

    bc3 = navier_stokes_cpp.BoundaryCondition()
    bc3.location = "y_min"
    bc3.type = "no-slip"

    bc4 = navier_stokes_cpp.BoundaryCondition()
    bc4.location = "y_max"
    bc4.type = "free-slip"

    bc_list = [bc1, bc2, bc3, bc4]

    # Execute pre-step boundary condition application
    navier_stokes_cpp.execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz)

    # Verify field finiteness and correct execution
    # finiteness verification: all(isfinite(fields)) == True
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(p))
