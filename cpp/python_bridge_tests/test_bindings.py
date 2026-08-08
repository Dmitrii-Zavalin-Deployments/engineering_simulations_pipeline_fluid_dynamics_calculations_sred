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
# NARRATIVE SECTION 1: Extension Module Availability and Introspection
# ============================================================================
# The pybind11 module must successfully load into the Python interpreter,
# registering high-performance C++ classes, docstrings, and py::arg metadata.
# ============================================================================

def test_module_initialization():
    assert navier_stokes_cpp is not None, "Extension module navier_stokes_cpp must be compiled and available."
    assert isinstance(navier_stokes_cpp.__doc__, str)
    assert len(navier_stokes_cpp.__doc__) > 0
    assert hasattr(navier_stokes_cpp, "NavierStokesSolver")
    assert hasattr(navier_stokes_cpp, "BoundaryCondition")


def test_signature_and_docstring_introspection():
    """Forces pybind11 to evaluate py::arg metadata for 100% binding coverage."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    # Inspect class and method docstrings to trigger py::arg formatting
    solver_doc = str(navier_stokes_cpp.NavierStokesSolver.__doc__)
    init_doc = str(navier_stokes_cpp.NavierStokesSolver.__init__.__doc__)
    step_doc = str(navier_stokes_cpp.NavierStokesSolver.step.__doc__)

    assert "Initialize solver grid dimensions" in solver_doc or "Initialize solver grid dimensions" in init_doc
    assert "Advance the Navier-Stokes system" in step_doc

    # Verify py::arg parameter names are registered in the signature string
    assert "nx" in init_doc or "nx" in solver_doc
    assert "fields" in step_doc


def test_boundary_condition_property_access():
    """Verifies read/write access to all BoundaryCondition fields."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    bc = navier_stokes_cpp.BoundaryCondition()
    bc.location = "x_min"
    bc.type = "inflow"
    bc.scalar_p = 101325.0
    bc.u_val = 1.5
    bc.v_val = 0.0
    bc.w_val = -0.5

    assert bc.location == "x_min"
    assert bc.type == "inflow"
    assert bc.scalar_p == 101325.0
    assert bc.u_val == 1.5
    assert bc.v_val == 0.0
    assert bc.w_val == -0.5


# ============================================================================
# NARRATIVE SECTION 2: Orchestrator Full Step Python Bridge Execution
# ============================================================================
# Test execution using both positional arguments and explicit keyword arguments 
# (kwargs) to trigger both argument-dispatch branches in pybind11.
# ============================================================================

def test_navier_stokes_solver_execution_positional():
    """Executes solver using positional arguments."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    dx, dy, dz = 0.1, 0.1, 0.1
    dt = 0.001
    density = 1000.0
    mu = 0.001

    fields = np.zeros((4, nx, ny, nz), dtype=np.float64)
    fields[0, :, :, :] = 0.1

    mask = np.ones((nx, ny, nz), dtype=np.int32)
    mask[0, :, :] = -1
    mask[-1, :, :] = -1

    fx = np.full((nx, ny, nz), 10.0, dtype=np.float64)
    fy = np.zeros((nx, ny, nz), dtype=np.float64)
    fz = np.zeros((nx, ny, nz), dtype=np.float64)

    bc = navier_stokes_cpp.BoundaryCondition()
    bc.location = "wall"
    bc.type = "no-slip"

    # Positional initialization
    solver = navier_stokes_cpp.NavierStokesSolver(
        nx, ny, nz, dx, dy, dz, 50, 1e-6, density
    )

    # Positional step execution
    solver.step(fields, mask, fx, fy, fz, [bc], dt, mu)

    assert np.all(np.isfinite(fields))


def test_navier_stokes_solver_execution_kwargs():
    """Executes solver using keyword arguments to trigger py::arg dispatching."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    dx, dy, dz = 0.1, 0.1, 0.1
    dt = 0.001
    density = 1000.0
    mu = 0.001

    fields = np.zeros((4, nx, ny, nz), dtype=np.float64)
    fields[0, :, :, :] = 0.1

    mask = np.ones((nx, ny, nz), dtype=np.int32)
    mask[0, :, :] = -1
    mask[-1, :, :] = -1

    fx = np.full((nx, ny, nz), 10.0, dtype=np.float64)
    fy = np.zeros((nx, ny, nz), dtype=np.float64)
    fz = np.zeros((nx, ny, nz), dtype=np.float64)

    bc = navier_stokes_cpp.BoundaryCondition()
    bc.location = "wall"
    bc.type = "no-slip"

    # Keyword argument initialization
    solver = navier_stokes_cpp.NavierStokesSolver(
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dy, dz=dz,
        max_poisson_iters=50,
        poisson_tolerance=1e-6,
        density=density
    )

    # Keyword argument step execution
    solver.step(
        fields=fields,
        mask=mask,
        fx=fx,
        fy=fy,
        fz=fz,
        bc_list=[bc],
        dt=dt,
        mu=mu
    )

    assert np.all(np.isfinite(fields))
    interior_u = fields[0, 1:nx-1, 1:ny-1, 1:nz-1]
    assert np.any(interior_u != 0.1)
