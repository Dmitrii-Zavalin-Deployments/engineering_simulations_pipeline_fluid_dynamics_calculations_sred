"""
@file test_bindings.py
@brief Literate Test Suite for Python Pybind11 Bindings Bridge & Orchestrator

This test file acts as a narrative document. Explanatory text and physical
principles are written as commented prose, while the executable Python assertions
verify correct interaction between the Python runtime and the C++ Navier-Stokes Orchestrator.
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
# registering the high-performance C++ classes and module documentation string.
# ============================================================================

def test_module_initialization():
    assert navier_stokes_cpp is not None, "Extension module navier_stokes_cpp must be compiled and available."
    assert isinstance(navier_stokes_cpp.__doc__, str)
    assert len(navier_stokes_cpp.__doc__) > 0
    assert hasattr(navier_stokes_cpp, "NavierStokesSolver")
    assert hasattr(navier_stokes_cpp, "BoundaryCondition")

# ============================================================================
# NARRATIVE SECTION 2: Orchestrator Full Step Python Bridge Execution
# ============================================================================
# The Python bridge function coordinates the full fractional-step Navier-Stokes 
# solver sequence (Pre-Step -> Predictor -> Pressure Poisson -> Corrector) 
# using 4D fields [u, v, w, p] and 3D domain masks.
# ============================================================================

def test_navier_stokes_solver_execution():
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    dx, dy, dz = 0.1, 0.1, 0.1
    dt = 0.001
    density = 1000.0
    mu = 0.001

    # Fields shape: (4, nx, ny, nz) -> [u, v, w, p]
    fields = np.zeros((4, nx, ny, nz), dtype=np.float64)
    fields[0, :, :, :] = 0.1  # Initial uniform u velocity

    # Domain mask: 1 = Fluid (Interior), -1 = Wall, 0 = Solid
    mask = np.ones((nx, ny, nz), dtype=np.int32)
    # Set outer boundaries to wall (-1)
    mask[0, :, :] = -1
    mask[-1, :, :] = -1
    mask[:, 0, :] = -1
    mask[:, -1, :] = -1
    mask[:, :, 0] = -1
    mask[:, :, -1] = -1

    fx = np.full((nx, ny, nz), 10.0, dtype=np.float64)
    fy = np.zeros((nx, ny, nz), dtype=np.float64)
    fz = np.zeros((nx, ny, nz), dtype=np.float64)

    # Configure Boundary Conditions (Testing all read/write attributes for 100% coverage)
    bc = navier_stokes_cpp.BoundaryCondition()
    bc.location = "wall"
    bc.type = "no-slip"
    bc.scalar_p = 0.0
    bc.u_val = 0.0
    bc.v_val = 0.0
    bc.w_val = 0.0

    bc_pressure = navier_stokes_cpp.BoundaryCondition()
    bc_pressure.location = "z_max"
    bc_pressure.type = "pressure"
    bc_pressure.scalar_p = 101325.0
    bc_pressure.u_val = 0.0
    bc_pressure.v_val = 0.0
    bc_pressure.w_val = 0.0

    bc_list = [bc, bc_pressure]

    # Initialize C++ Solver Orchestrator (Testing keyword arguments for 100% coverage)
    solver = navier_stokes_cpp.NavierStokesSolver(
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dy, dz=dz,
        max_poisson_iters=50,
        poisson_tolerance=1e-6,
        density=density
    )

    # Execute one time-step (Testing keyword arguments for step() coverage)
    solver.step(
        fields=fields,
        mask=mask,
        fx=fx,
        fy=fy,
        fz=fz,
        bc_list=bc_list,
        dt=dt,
        mu=mu
    )

    # Verify fields remain finite and valid after step execution
    assert np.all(np.isfinite(fields))
    
    # Assert that active interior fluid cells (mask == 1) have been updated by the solver pipeline
    interior_u = fields[0, 1:nx-1, 1:ny-1, 1:nz-1]
    assert np.any(interior_u != 0.1)
