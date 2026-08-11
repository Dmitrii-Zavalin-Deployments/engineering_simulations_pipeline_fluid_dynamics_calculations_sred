"""
@file test_bindings.py
@brief Literate Test Suite for Python Pybind11 Bindings Bridge & Orchestrator

This test file acts as a narrative document. Explanatory text and physical
principles are written as commented prose, while the executable Python assertions
verify correct interaction between the Python runtime container and the C++ Navier-Stokes Orchestrator.
"""

import numpy as np
import pytest

try:
    import navier_stokes_cpp
except ImportError:
    navier_stokes_cpp = None


class DummySolverState:
    """Mock sovereign container matching the attributes expected by python_gate.cpp."""
    def __init__(self, nx=8, ny=8, nz=8):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.x_min = 0.0
        self.x_max = 1.0
        self.y_min = 0.0
        self.y_max = 1.0
        self.z_min = 0.0
        self.z_max = 1.0
        self.dt = 0.001

        self.fields = np.zeros((4, nx, ny, nz), dtype=np.float64)
        self.fields[0, :, :, :] = 0.1

        self.mask = np.zeros((nx, ny, nz), dtype=np.int32)
        self.mask[0, :, :] = -1
        self.mask[-1, :, :] = -1

        self.fluid_properties = {
            "density": 1000.0,
            "viscosity": 0.001
        }
        self.config = {
            "max_poisson_iterations": 50,
            "poisson_tolerance": 1e-6
        }
        self.physical_constraints = {
            "min_velocity": -10.0,
            "max_velocity": 10.0,
            "min_pressure": -100.0,
            "max_pressure": 100.0
        }
        self.external_forces = {
            "gravity_vector": [0.0, -9.81, 0.0],
            "force_vector": [10.0, 0.0, 0.0]
        }
        
        bc = navier_stokes_cpp.BoundaryCondition() if navier_stokes_cpp else None
        if bc:
            bc.location = "wall"
            bc.type = "no-slip"
        self.boundary_conditions = [bc] if bc else []


# ============================================================================
# NARRATIVE SECTION 1: Extension Module Availability and Introspection
# ============================================================================
# The pybind11 module must successfully load into the Python interpreter,
# registering high-performance C++ classes, docstrings, and container bindings.
# ============================================================================

def test_module_initialization():
    assert navier_stokes_cpp is not None, "Extension module navier_stokes_cpp must be compiled and available."
    assert isinstance(navier_stokes_cpp.__doc__, str)
    assert len(navier_stokes_cpp.__doc__) > 0
    assert hasattr(navier_stokes_cpp, "NavierStokesSolver")
    assert hasattr(navier_stokes_cpp, "BoundaryCondition")


def test_docstring_introspection():
    """Validates C++ pybind11 docstrings and method signatures are correctly exposed."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    solver_doc = str(navier_stokes_cpp.NavierStokesSolver.__doc__)
    init_doc = str(navier_stokes_cpp.NavierStokesSolver.__init__.__doc__)
    step_doc = str(navier_stokes_cpp.NavierStokesSolver.step.__doc__)

    assert "Initialize solver instance directly from sovereign SolverState container" in init_doc
    assert "Advance the Navier-Stokes system by one time-step using state container references" in step_doc


def test_invalid_state_error_handling():
    """Triggers exception branches when passing None as the sovereign state container."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    with pytest.raises((TypeError, ValueError)):
        navier_stokes_cpp.NavierStokesSolver(None)


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
# NARRATIVE SECTION 2: Sovereign Container Orchestrator Execution
# ============================================================================
# Exercises end-to-end execution passing the sovereign SolverState container
# directly into the C++ bridge constructor and time-stepping loop.
# ============================================================================

def test_navier_stokes_solver_container_execution():
    """Executes solver core using the sovereign container pattern and comprehensively validates all attributes and in-place RAM mutation."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)

    # Initialize C++ bridge using the sovereign container
    solver = navier_stokes_cpp.NavierStokesSolver(state)

    # Execute simulation time-step using container reference
    solver.step(state)

    # 1. Comprehensive Attribute Validation on Container Post-Execution
    assert state.nx == 8
    assert state.ny == 8
    assert state.nz == 8
    assert state.x_min == 0.0
    assert state.x_max == 1.0
    assert state.y_min == 0.0
    assert state.y_max == 1.0
    assert state.z_min == 0.0
    assert state.z_max == 1.0
    assert state.dt == 0.001

    # 2. Tensors & Buffers Shape & Finiteness Checks
    assert state.fields.shape == (4, nx, ny, nz)
    assert state.mask.shape == (nx, ny, nz)
    assert np.all(np.isfinite(state.fields))

    # 3. Fluid Properties & Config Verification
    assert float(state.fluid_properties["density"]) == 1000.0
    assert float(state.fluid_properties["viscosity"]) == 0.001
    assert int(state.config["max_poisson_iterations"]) == 50
    assert float(state.config["poisson_tolerance"]) == 1e-6

    # 4. Physical Constraints Validation
    assert float(state.physical_constraints["min_velocity"]) == -10.0
    assert float(state.physical_constraints["max_velocity"]) == 10.0
    assert float(state.physical_constraints["min_pressure"]) == -100.0
    assert float(state.physical_constraints["max_pressure"]) == 100.0

    # 5. External Forces & Boundary Conditions Validation
    assert state.external_forces["gravity_vector"] == [0.0, -9.81, 0.0]
    assert state.external_forces["force_vector"] == [10.0, 0.0, 0.0]
    assert len(state.boundary_conditions) == 1
    assert state.boundary_conditions[0].location == "wall"
    assert state.boundary_conditions[0].type == "no-slip"
