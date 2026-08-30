"""
@file test_python_gate_coverage.py
@brief Comprehensive Python unit test suite targeting 100% test coverage for python_gate.cpp.

WHAT: This test suite provides exhaustive verification of the Pybind11 bridge layer (python_gate.cpp) 
that connects the Python sovereign SolverState container with the high-performance C++ Navier-Stokes Orchestrator.

HOW: It constructs valid and malformed mock solver states, passing them through the C++ constructor, 
time-stepping execution loops, and field synchronization routines to verify strict contract enforcement, 
exception translation, tensor shape validation, and boundary condition parsing.

WHY: Achieving 100% test coverage ensures that all physical bounds, geometry specifications, data type casting 
safeguards, and numerical stability checks fail gracefully with informative, predictable exceptions 
rather than causing segmentation faults or silent memory corruption during computational fluid dynamics simulations.
"""

import pytest
import numpy as np
import navier_stokes_cpp as nsc


class MockSolverState:
    """
    We define a foundational mock solver state representing a sovereign Python container 
    holding domain dimensions, physical properties, solver configurations, and tensor buffers.
    """
    def __init__(
        self,
        nx=3,
        ny=3,
        nz=3,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        density=1000.0,
        viscosity=0.01,
        dt=0.01,
        max_iters=10,
        tol=1e-6,
    ):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.fluid_properties = {"density": density, "viscosity": viscosity}
        self.config = {
            "max_poisson_iterations": max_iters,
            "poisson_tolerance": tol,
        }
        self.dt = dt
        self.fields = np.zeros((4, nx, ny, nz), dtype=np.float64)
        self.mask = np.ones((nx, ny, nz), dtype=np.int32)
        self.external_forces = {
            "gravity_vector": [0.0, 0.0, -9.81],
            "force_vector": [0.0, 0.0, 0.0],
        }
        self.boundary_conditions = []


def test_constructor_none_state():
    """
    A null or None state object violates the fundamental initialization contract 
    and must immediately raise a value error.
    """
    with pytest.raises(ValueError, match="state object cannot be None"):
        nsc.NavierStokesSolver(None)


def test_constructor_invalid_geometry():
    """
    Node-based discretization requires at least 2 nodes per dimension (nx, ny, nz >= 2) 
    to compute valid spatial grid spacing (dx, dy, dz).
    """
    state = MockSolverState(nx=1)
    with pytest.raises(ValueError, match="GEOMETRY ERROR"):
        nsc.NavierStokesSolver(state)


def test_constructor_invalid_spacing():
    """
    Zero or non-finite spatial spans result in invalid grid spacing values, 
    triggering strict geometry error protection.
    """
    state = MockSolverState(x_min=1.0, x_max=1.0)
    with pytest.raises(ValueError, match="GEOMETRY ERROR"):
        nsc.NavierStokesSolver(state)

    state_nan = MockSolverState(x_max=float("nan"))
    with pytest.raises(ValueError, match="GEOMETRY ERROR"):
        nsc.NavierStokesSolver(state_nan)


def test_constructor_invalid_density():
    """
    Fluid density must remain strictly positive and finite to maintain physical validity 
    within the Navier-Stokes momentum equations.
    """
    state = MockSolverState(density=0.0)
    with pytest.raises(ValueError, match="PHYSICS ERROR"):
        nsc.NavierStokesSolver(state)

    state_inf = MockSolverState(density=float("inf"))
    with pytest.raises(ValueError, match="PHYSICS ERROR"):
        nsc.NavierStokesSolver(state_inf)


def test_constructor_missing_attributes():
    """
    Incomplete state containers lacking required attributes must be caught and wrapped 
    into a standard state contract error.
    """
    class IncompleteState:
        pass

    with pytest.raises(
        ValueError, match="STATE CONTRACT ERROR|Missing or invalid attributes"
    ):
        nsc.NavierStokesSolver(IncompleteState())


def test_constructor_type_error_rethrow():
    """
    Type mismatches during attribute casting (e.g., string assigned to integer grid dimension) 
    must translate cleanly into pybind11 type errors.
    """
    class BadTypeState:
        nx = "not_an_int"
        ny = 3
        nz = 3
        x_min = 0.0
        x_max = 1.0
        y_min = 0.0
        y_max = 1.0
        z_min = 0.0
        z_max = 1.0
        fluid_properties = {"density": 1000.0}
        config = {"max_poisson_iterations": 10, "poisson_tolerance": 1e-6}

    with pytest.raises((ValueError, TypeError)):
        nsc.NavierStokesSolver(BadTypeState())


def test_step_none_state():
    """
    Executing a time-step with a None state reference must raise a fatal value error.
    """
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)
    with pytest.raises(ValueError, match="state object cannot be None"):
        solver.step(None)


def test_step_invalid_dt():
    """
    Temporal increments (dt) must be strictly positive and finite. Non-positive or NaN 
    time steps trigger a temporal error.
    """
    state = MockSolverState(dt=0.0)
    solver = nsc.NavierStokesSolver(state)
    with pytest.raises(ValueError, match="TEMPORAL ERROR"):
        solver.step(state)

    state.dt = float("nan")
    with pytest.raises(ValueError, match="TEMPORAL ERROR"):
        solver.step(state)


def test_step_invalid_viscosity():
    """
    Dynamic viscosity cannot be negative or non-finite.
    """
    state = MockSolverState(viscosity=-0.01)
    solver = nsc.NavierStokesSolver(state)
    with pytest.raises(ValueError, match="PHYSICS ERROR"):
        solver.step(state)

    state.fluid_properties["viscosity"] = float("nan")
    with pytest.raises(ValueError, match="PHYSICS ERROR"):
        solver.step(state)


def test_step_1d_mask():
    """
    The solver supports both 3D volumetric and flattened 1D array masks for domain cutout mapping.
    """
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)
    state.mask = np.ones(state.nx * state.ny * state.nz, dtype=np.int32)
    solver.step(state)


def test_step_invalid_mask_ndim():
    """
    Unsupported mask dimensions (e.g., 2D matrices) must be rejected with a geometry error.
    """
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)
    state.mask = np.ones((state.nx, state.ny), dtype=np.int32)
    with pytest.raises(ValueError, match="GEOMETRY ERROR"):
        solver.step(state)


def test_step_invalid_mask_ndim_2():
    """
    Additional 2D mask dimension check branch to ensure complete code path coverage in python_gate.cpp.
    """
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)
    state.mask = np.zeros((4, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="GEOMETRY ERROR: mask must be a 1D or 3D NumPy array"):
        solver.step(state)


def test_step_boundary_conditions_dict_parsing():
    """
    Boundary conditions supplied via dictionary configurations are correctly parsed and mapped 
    into C++ BoundaryCondition structures.
    """
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)
    state.boundary_conditions = [
        {
            "location": "x_min",
            "type": "dirichlet",
            "values": {"u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0},
        }
    ]
    solver.step(state)


def test_step_boundary_conditions_non_finite_values():
    """
    Non-finite boundary condition values (NaN or inf) within dictionaries trigger an advection 
    term explosion exception to preserve numerical stability.
    """
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)

    state.boundary_conditions = [{
        "location": "x_min",
        "type": "dirichlet",
        "values": {"u": float("nan")}
    }]
    with pytest.raises(RuntimeError, match="Advection term exploded"):
        solver.step(state)

    state.boundary_conditions = [{
        "location": "x_min",
        "type": "dirichlet",
        "values": {"v": float("inf")}
    }]
    with pytest.raises(RuntimeError, match="Advection term exploded"):
        solver.step(state)

    state.boundary_conditions = [{
        "location": "x_min",
        "type": "dirichlet",
        "values": {"w": float("nan")}
    }]
    with pytest.raises(RuntimeError, match="Advection term exploded"):
        solver.step(state)

    state.boundary_conditions = [{
        "location": "x_min",
        "type": "dirichlet",
        "values": {"p": float("inf")}
    }]
    with pytest.raises(RuntimeError, match="Advection term exploded"):
        solver.step(state)


def test_step_boundary_condition_object_non_finite():
    """
    Direct Cosphere BoundaryCondition instances containing non-finite values correctly trigger 
    runtime safety exceptions.
    """
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)

    bc = nsc.BoundaryCondition()
    bc.location = "x_min"
    bc.type = "dirichlet"
    bc.u_val = float("nan")

    state.boundary_conditions = [bc]
    with pytest.raises(RuntimeError, match="Advection term exploded"):
        solver.step(state)


def test_step_boundary_condition_object_instance():
    """
    Direct C++ BoundaryCondition object instances passed inside the boundary conditions list 
    are correctly processed without dictionary wrappers.
    """
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)

    bc = nsc.BoundaryCondition()
    bc.location = "outlet"
    bc.type = "neumann"
    bc.u_val = 0.0
    bc.v_val = 0.0
    bc.w_val = 0.0
    bc.scalar_p = 0.0
    state.boundary_conditions = [bc]

    solver.step(state)


def test_sync_fields_none_state():
    """
    Synchronizing fields with a None state reference must raise a fatal value error.
    """
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)
    with pytest.raises(ValueError, match="state object cannot be None"):
        solver.sync_fields(None)