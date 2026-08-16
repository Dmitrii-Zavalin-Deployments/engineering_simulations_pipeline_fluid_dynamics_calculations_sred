"""
@file test_cpp_gate.py
@brief Literate Test Suite for C++ Gate Bridge Module (src/cpp_gate.py)

This test module acts as a narrative document and exhaustive verification suite for src/cpp_gate.py.
Explanatory text and physical equations are written as commented prose, while executable Python
assertions verify zero-copy memory binding, boundary condition parsing, time-step metric tracking,
fallback resolution, and exception handling across all execution paths.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from src import cpp_gate
from src.cpp_gate import (
    _convert_boundary_conditions,
    _dict_to_boundary_condition,
    _get_or_create_cpp_solver,
    step_simulation,
)


class DummyState:
    """Mock sovereign state container matching SolverState interface attributes."""

    def __init__(self, dt=0.001):
        self.dt = dt
        self.current_iteration = 0
        self.current_time = 0.0
        self.boundary_conditions = []
        self.input_data = {}
        self._cpp_solver = None


# ============================================================================
# NARRATIVE SECTION 1: Boundary Condition Conversion & Primitive Mapping
# ============================================================================
# Surface boundary conditions prescribe field values on 3D spatial boundaries:
#     u = u_bc,  v = v_bc,  w = w_bc,  p = p_bc
#
# The bridge converts Python dictionary definitions into statically typed C++
# BoundaryCondition structures. Values dictionary mapping obeys:
#     "u" -> bc_obj.u_val
#     "v" -> bc_obj.v_val
#     "w" -> bc_obj.w_val
#     "p" -> bc_obj.scalar_p
# ============================================================================


def test_dict_to_boundary_condition_p_mapping():
    """Verifies complete conversion of boundary dictionary fields including pressure scalar p."""
    # We define a boundary condition dictionary containing velocity components and pressure:
    bc_dict = {
        "location": "x_min",
        "type": "inflow",
        "values": {
            "u": 1.5,
            "v": -0.5,
            "w": 0.25,
            "p": 101325.0,
        },
    }

    # We execute boundary condition conversion:
    bc_obj = _dict_to_boundary_condition(bc_dict)

    # The resulting C++ object must correctly reflect all field attributes:
    assert bc_obj.location == "x_min"
    assert bc_obj.type == "inflow"
    assert abs(bc_obj.u_val - 1.5) < 1e-9
    assert abs(bc_obj.v_val - (-0.5)) < 1e-9
    assert abs(bc_obj.w_val - 0.25) < 1e-9
    assert abs(bc_obj.scalar_p - 101325.0) < 1e-9


def test_convert_boundary_conditions_in_place():
    """Verifies list conversion preserving pre-existing BoundaryCondition objects alongside dicts."""
    state = DummyState()

    # Pre-instantiated C++ BoundaryCondition object:
    import navier_stokes_cpp
    existing_bc = navier_stokes_cpp.BoundaryCondition()
    existing_bc.location = "wall"

    # Hybrid list of dictionary and C++ object:
    state.boundary_conditions = [
        {"location": "y_max", "type": "no-slip", "values": {"u": 0.0}},
        existing_bc,
    ]

    _convert_boundary_conditions(state)

    # Both elements must now be instances of C++ BoundaryCondition:
    assert len(state.boundary_conditions) == 2
    assert isinstance(state.boundary_conditions[0], navier_stokes_cpp.BoundaryCondition)
    assert state.boundary_conditions[0].location == "y_max"
    assert state.boundary_conditions[1] is existing_bc


# ============================================================================
# NARRATIVE SECTION 2: Engine Initialization & Sovereign Memory Binding
# ============================================================================
# The C++ solver engine is bound lazily to the sovereign state container:
#     state._cpp_solver = navier_stokes_cpp.NavierStokesSolver(state)
#
# Null container references are strictly forbidden under the non-default policy:
#     state == None ==> raise ValueError
# ============================================================================


def test_get_or_create_cpp_solver_none_state():
    """Verifies that passing None state to _get_or_create_cpp_solver triggers a ValueError."""
    with pytest.raises(ValueError, match="state must be explicitly provided"):
        _get_or_create_cpp_solver(None)


def test_get_or_create_cpp_solver_reuse_existing():
    """Verifies that existing bound C++ solver instances are reused without re-initialization."""
    state = DummyState()
    mock_solver = MagicMock()
    state._cpp_solver = mock_solver

    # Retrieving engine when _cpp_solver is already attached must return the same reference:
    solver = _get_or_create_cpp_solver(state)
    assert solver is mock_solver


# ============================================================================
# NARRATIVE SECTION 3: Discrete Time Integration & Metric Propagation
# ============================================================================
# Each successful simulation step updates temporal metrics:
#     t_{n+1} = t_n + dt
#     n_{next} = n + 1
#
# Primary dt resolution evaluates float(state.dt). If missing or invalid,
# fallback extraction reads state.input_data["simulation_parameters"]["time_step"].
# If both fail, a KeyError is raised.
# ============================================================================


def test_step_simulation_happy_path():
    """Verifies end-to-end simulation time integration and metric advancement."""
    state = DummyState(dt=0.005)

    mock_solver = MagicMock()
    mock_solver.step = MagicMock()
    mock_solver.sync_fields = MagicMock()
    state._cpp_solver = mock_solver

    # Execute simulation step:
    step_simulation(state)

    # Assert C++ step and field synchronization were executed:
    mock_solver.step.assert_called_once_with(state)
    mock_solver.sync_fields.assert_called_once_with(state)

    # Verify state tracking metrics:
    #     n = 0 + 1 = 1
    #     t = 0.0 + 0.005 = 0.005
    assert state.current_iteration == 1
    assert abs(state.current_time - 0.005) < 1e-9


def test_step_simulation_dt_fallback_resolution():
    """Verifies fallback extraction of time_step from state.input_data when state.dt is invalid."""
    state = DummyState(dt=None)  # Triggers AttributeError/TypeError on float(state.dt)
    state.input_data = {"simulation_parameters": {"time_step": 0.0025}}

    mock_solver = MagicMock()
    mock_solver.step = MagicMock()
    mock_solver.sync_fields = MagicMock()
    state._cpp_solver = mock_solver

    step_simulation(state)

    # Verify metric increment using fallback time step dt = 0.0025:
    assert state.current_iteration == 1
    assert abs(state.current_time - 0.0025) < 1e-9


def test_step_simulation_missing_dt_raises_key_error():
    """Verifies KeyError when both state.dt and state.input_data time_step sources are missing."""
    state = DummyState(dt=None)
    state.input_data = {}  # No simulation_parameters key

    mock_solver = MagicMock()
    mock_solver.step = MagicMock()
    mock_solver.sync_fields = MagicMock()
    state._cpp_solver = mock_solver

    with pytest.raises(KeyError, match="Simulation time step 'dt' or 'simulation_parameters.time_step'"):
        step_simulation(state)


# ============================================================================
# NARRATIVE SECTION 4: Exception Handling & Bridge Robustness
# ============================================================================
# Missing solver contracts or C++ execution failures must be caught, logged,
# and re-raised as descriptive RuntimeError exceptions.
# ============================================================================


def test_step_simulation_none_state():
    """Verifies that passing None state to step_simulation raises ValueError."""
    with pytest.raises(ValueError, match="state must be explicitly provided"):
        step_simulation(None)


def test_step_simulation_missing_sync_fields():
    """Verifies RuntimeError when bound C++ solver lacks required callable sync_fields method."""
    state = DummyState()

    class SolverWithoutSync:
        def step(self, s):
            pass

    state._cpp_solver = SolverWithoutSync()

    with pytest.raises(RuntimeError, match="missing required callable 'sync_fields' method"):
        step_simulation(state)


def test_step_simulation_execution_failure():
    """Verifies that underlying C++ execution exceptions are wrapped in RuntimeError."""
    state = DummyState()

    mock_solver = MagicMock()
    mock_solver.step.side_effect = ValueError("Advection term exploded")
    state._cpp_solver = mock_solver

    with pytest.raises(RuntimeError, match=r"C\+\+ execution failure during solver step"):
        step_simulation(state)


def test_cpp_gate_import_error_handling():
    """Verifies descriptive ImportError when compiled C++ module navier_stokes_cpp is missing."""
    with patch.dict(sys.modules, {"navier_stokes_cpp": None}):
        with pytest.raises(ImportError, match=r"Failed to import compiled C\+\+ module 'navier_stokes_cpp'"):
            importlib.reload(cpp_gate)

    # Restore module state post-test
    importlib.reload(cpp_gate)
