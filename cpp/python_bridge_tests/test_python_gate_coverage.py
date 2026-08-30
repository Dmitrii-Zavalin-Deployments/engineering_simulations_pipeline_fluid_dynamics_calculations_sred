"""
@file test_python_gate_coverage.py
@brief Comprehensive Python unit test suite targeting 100% test coverage for python_gate.cpp.
"""

import pytest
import numpy as np
import navier_stokes_cpp as nsc


class MockSolverState:
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
    with pytest.raises(ValueError, match="state object cannot be None"):
        nsc.NavierStokesSolver(None)


def test_constructor_invalid_geometry():
    state = MockSolverState(nx=1)
    with pytest.raises(ValueError, match="GEOMETRY ERROR"):
        nsc.NavierStokesSolver(state)


def test_constructor_invalid_spacing():
    state = MockSolverState(x_min=1.0, x_max=1.0)
    with pytest.raises(ValueError, match="GEOMETRY ERROR"):
        nsc.NavierStokesSolver(state)

    state_nan = MockSolverState(x_max=float("nan"))
    with pytest.raises(ValueError, match="GEOMETRY ERROR"):
        nsc.NavierStokesSolver(state_nan)


def test_constructor_invalid_density():
    state = MockSolverState(density=0.0)
    with pytest.raises(ValueError, match="PHYSICS ERROR"):
        nsc.NavierStokesSolver(state)

    state_inf = MockSolverState(density=float("inf"))
    with pytest.raises(ValueError, match="PHYSICS ERROR"):
        nsc.NavierStokesSolver(state_inf)


def test_constructor_missing_attributes():
    class IncompleteState:
        pass

    with pytest.raises(
        ValueError, match="STATE CONTRACT ERROR|Missing or invalid attributes"
    ):
        nsc.NavierStokesSolver(IncompleteState())


def test_constructor_type_error_rethrow():
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
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)
    with pytest.raises(ValueError, match="state object cannot be None"):
        solver.step(None)


def test_step_invalid_dt():
    state = MockSolverState(dt=0.0)
    solver = nsc.NavierStokesSolver(state)
    with pytest.raises(ValueError, match="TEMPORAL ERROR"):
        solver.step(state)

    state.dt = float("nan")
    with pytest.raises(ValueError, match="TEMPORAL ERROR"):
        solver.step(state)


def test_step_invalid_viscosity():
    state = MockSolverState(viscosity=-0.01)
    solver = nsc.NavierStokesSolver(state)
    with pytest.raises(ValueError, match="PHYSICS ERROR"):
        solver.step(state)

    state.fluid_properties["viscosity"] = float("nan")
    with pytest.raises(ValueError, match="PHYSICS ERROR"):
        solver.step(state)


def test_step_1d_mask():
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)
    state.mask = np.ones(state.nx * state.ny * state.nz, dtype=np.int32)
    solver.step(state)


def test_step_invalid_mask_ndim():
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)
    state.mask = np.ones((state.nx, state.ny), dtype=np.int32)
    with pytest.raises(ValueError, match="GEOMETRY ERROR"):
        solver.step(state)


def test_step_boundary_conditions_dict_parsing():
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
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)

    bc = nsc.BoundaryCondition()
    bc.location = "x_min"
    bc.type = "dirichlet"
    bc.u_val = float("nan")

    state.boundary_conditions = [bc]
    with pytest.raises(RuntimeError, match="Advection term exploded"):
        solver.step(state)


def test_sync_fields_none_state():
    state = MockSolverState()
    solver = nsc.NavierStokesSolver(state)
    with pytest.raises(ValueError, match="state object cannot be None"):
        solver.sync_fields(None)
