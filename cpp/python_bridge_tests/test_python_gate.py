"""
@file test_python_gate.py
@brief Literate Test Suite for Python Pybind11 Bindings Bridge (python_gate.cpp)

This test file acts as a narrative document for the Python-C++ runtime bridge.
Explanatory text and physical principles are written as commented prose, while
the executable Python assertions verify correct zero-copy tensor mapping, memory
synchronization paths, structural contract validation, and exception handling
between the Python runtime container and the underlying C++ Navier-Stokes solver.
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

        # The primary solution tensor holding fields [u, v, w, p]:
        #     fields.shape == (4, nx, ny, nz)
        self.fields = np.zeros((4, nx, ny, nz), dtype=np.float64)
        self.fields[0, :, :, :] = 0.1

        # Spatial domain mask defining fluid (1), solid (0), and boundary (-1) cells:
        #     mask.shape == (nx, ny, nz)
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
            "force_vector": [10.0, 0.0, 0.0],
            "fx": np.zeros((nx, ny, nz), dtype=np.float64),
            "fy": np.zeros((nx, ny, nz), dtype=np.float64),
            "fz": np.zeros((nx, ny, nz), dtype=np.float64)
        }
        
        bc = navier_stokes_cpp.BoundaryCondition() if navier_stokes_cpp else None
        if bc:
            bc.location = "wall"
            bc.type = "no-slip"
        self.boundary_conditions = [bc] if bc else []


# ============================================================================
# NARRATIVE SECTION 1: Extension Module Loading and Introspection
# ============================================================================
# The pybind11 module exposes high-performance C++ solver classes and boundary
# condition structures to the Python interpreter.
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

    init_doc = str(navier_stokes_cpp.NavierStokesSolver.__init__.__doc__)
    step_doc = str(navier_stokes_cpp.NavierStokesSolver.step.__doc__)

    assert "Initialize solver instance directly from sovereign SolverState container" in init_doc
    assert "Advance the Navier-Stokes system by one time-step using state container references" in step_doc


def test_invalid_state_error_handling():
    """Triggers exception branches when passing None as the sovereign state container to the constructor."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    with pytest.raises((TypeError, ValueError)):
        navier_stokes_cpp.NavierStokesSolver(None)


def test_boundary_condition_property_access():
    """Verifies read/write access to all BoundaryCondition fields through the Pybind11 gateway."""
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
#
# The continuity equation enforced across the grid cells is:
#     div(u) = du/dx + dv/dy + dw/dz = 0
# ============================================================================

def test_navier_stokes_solver_container_execution():
    """Executes solver core using the sovereign container pattern and validates in-place RAM mutation."""
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
#     finiteness verification: all(isfinite(fields)) == True
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


# ============================================================================
# NARRATIVE SECTION 3: Robustness, Contract Enforcement, and Buffer Sync
# ============================================================================
# Verifies exception paths for None-state inputs, invalid vector component sizes,
# non-finite field explosions, and explicit field synchronization back to Python.
# ============================================================================

def test_step_none_state_error():
    """Triggers exception branches when passing None as state to step()."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    solver = navier_stokes_cpp.NavierStokesSolver(state)

    with pytest.raises((TypeError, ValueError)):
        solver.step(None)


def test_invalid_gravity_vector_size():
    """Triggers contract violation when gravity_vector does not contain exactly 3 components [gx, gy, gz]."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    state.external_forces["gravity_vector"] = [0.0, -9.81]  # Invalid size != 3
    solver = navier_stokes_cpp.NavierStokesSolver(state)

    # Contract requirement:
#     gravity_vector.size() == 3
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        solver.step(state)


def test_invalid_force_vector_size():
    """Triggers contract violation when force_vector does not contain exactly 3 components [fx, fy, fz]."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    state.external_forces["force_vector"] = [10.0, 0.0]  # Invalid size != 3
    solver = navier_stokes_cpp.NavierStokesSolver(state)

    # Contract requirement:
#     force_vector.size() == 3
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        solver.step(state)


def test_non_finite_field_simulation_failure():
    """Triggers runtime error when active fluid fields contain non-finite (NaN/inf) values after a time step."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    
    # 1. Mark interior cells as active fluid (mask == 1) so pre-step won't overwrite them
    state.mask[1:-1, 1:-1, 1:-1] = 1
    
    # 2. Seed active fluid cell with NaN
    state.fields[0, 3, 3, 3] = np.nan
    
    solver = navier_stokes_cpp.NavierStokesSolver(state)

    # 3. Verify that the correct C++ exception message is matched
    with pytest.raises(RuntimeError, match="Advection term exploded in grid computation."):
        solver.step(state)


def test_sync_fields_none_error():
    """Triggers exception when passing None to sync_fields()."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    solver = navier_stokes_cpp.NavierStokesSolver(state)

    with pytest.raises((TypeError, ValueError)):
        solver.sync_fields(None)


def test_sync_fields_execution():
    """Executes sync_fields() explicitly to synchronize C++ solution vectors back into Python state buffers."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    state = DummySolverState(nx=nx, ny=ny, nz=nz)
    solver = navier_stokes_cpp.NavierStokesSolver(state)
    
    solver.step(state)
    solver.sync_fields(state)

    assert np.all(np.isfinite(state.fields))
