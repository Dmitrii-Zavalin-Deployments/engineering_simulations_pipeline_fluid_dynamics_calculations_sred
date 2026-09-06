"""
test_state.py
This test module serves as a narrative document and verification suite for src/state.py.
Explanatory text and physical discretization formulas are written as commented prose,
while executable Python assertions verify state instantiation, grid resolution steps,
zero-copy memory views, unconstrained physical evolution, and exception handling across all paths,
aligned with the compiled C++ core engine architecture (python_gate.cpp, simulation_prestep.cpp, orchestrator.cpp).
"""

import numpy as np
import pytest

from src.state import SolverState


def _create_valid_state_inputs():
    """Generates valid minimal input_data and config_data structures."""
    input_data = {
        "grid": {
            "nx": 4,
            "ny": 4,
            "nz": 4,
            "x_min": 0.0,
            "x_max": 2.0,
            "y_min": 0.0,
            "y_max": 2.0,
            "z_min": 0.0,
            "z_max": 2.0,
        },
        "fluid_properties": {"density": 1000.0, "viscosity": 0.001},
        "initial_conditions": {
            "velocity": [1.0, 2.0, 3.0],
            "pressure": 101325.0,
        },
        "simulation_parameters": {
            "time_step": 0.01,
            "total_time": 0.1,
            "output_interval": 2,
        },
        "boundary_conditions": [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0}}
        ],
        "external_forces": {"gravity": [0.0, -9.81, 0.0]},
        "domain_configuration": {"type": "box"},
        "physical_constraints": {
            "min_velocity": -50.0,
            "max_velocity": 50.0,
            "min_pressure": 0.0,
            "max_pressure": 200000.0,
        },
        "mask": [1] * 64,  # nx * ny * nz = 4 * 4 * 4 = 64
    }

    config_data = {"poisson_solver": "cg", "tolerance": 1e-6}

    return input_data, config_data


# ============================================================================
# NARRATIVE SECTION 1: Operational State Construction & Zero-Copy Views
# ============================================================================
# Grid spatial step sizes are computed along each cartesian coordinate axis:
#     dx = (x_max - x_min) / nx
#     dy = (y_max - y_min) / ny
#     dz = (z_max - z_min) / nz
#
# Total required time steps are calculated from physical time extents:
#     N_iterations = round(total_time / dt)
#
# 4D fields allocation follows shape (4, nx, ny, nz) in C-contiguous memory.
# Memory slices share views with parent fields array:
#     u = fields[0],  v = fields[1],  w = fields[2],  p = fields[3]
# ============================================================================


def test_solver_state_initialization_success():
    """Verifies complete state initialization, spatial step math, zero-copy memory binding, and C++ pre-step field population."""
    input_data, config_data = _create_valid_state_inputs()

    # Spatial step math evaluation:
    #     dx = (2.0 - 0.0) / 4 = 0.5 m
    #     dy = (2.0 - 0.0) / 4 = 0.5 m
    #     dz = (2.0 - 0.0) / 4 = 0.5 m
    #     N_iterations = round(0.1 / 0.01) = 10
    state = SolverState(input_data, config_data)

    assert abs(state.dx - 0.5) < 1e-9
    assert abs(state.dy - 0.5) < 1e-9
    assert abs(state.dz - 0.5) < 1e-9
    assert state.total_iterations == 10

    # Zero-copy memory view assertions:
    assert state.u.base is state.fields
    assert state.v.base is state.fields
    assert state.w.base is state.fields
    assert state.p.base is state.fields

    # Execute C++ pre-step / cold start to populate initial condition fields
    if hasattr(state, "execute_pre_step"):
        state.execute_pre_step()

    # Initial condition field population assertion after C++ pre-step execution:
    assert np.allclose(state.u, 1.0)
    assert np.allclose(state.v, 2.0)
    assert np.allclose(state.w, 3.0)
    assert np.allclose(state.p, 101325.0)


# ============================================================================
# NARRATIVE SECTION 2: Schema Non-Default Input Policy Verification
# ============================================================================
# Strict state construction forbids null arguments, missing grid parameters,
# missing sub-schema sections, or malformed velocity sequences.
# ============================================================================


def test_solver_state_null_input_or_config():
    """Verifies exception handling when input_data or config_data is None."""
    input_data, config_data = _create_valid_state_inputs()

    with pytest.raises((ValueError, RuntimeError)):
        SolverState(None, config_data)

    with pytest.raises((ValueError, RuntimeError)):
        SolverState(input_data, None)


def test_solver_state_missing_grid_section_or_keys():
    """Verifies exception handling when grid section or required grid keys are missing/None."""
    input_data, config_data = _create_valid_state_inputs()

    # Missing 'grid' section:
    del input_data["grid"]
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        SolverState(input_data, config_data)

    # None 'grid' section:
    input_data, config_data = _create_valid_state_inputs()
    input_data["grid"] = None
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        SolverState(input_data, config_data)

    # Missing key inside grid:
    input_data, config_data = _create_valid_state_inputs()
    input_data["grid"]["nx"] = None
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        SolverState(input_data, config_data)


def test_solver_state_missing_subsections():
    """Verifies exception handling when required schema sections are missing or set to None."""
    required_sections = [
        "fluid_properties",
        "initial_conditions",
        "simulation_parameters",
        "boundary_conditions",
        "external_forces",
        "domain_configuration",
        "physical_constraints",
        "mask",
    ]

    for sec in required_sections:
        input_data, config_data = _create_valid_state_inputs()
        del input_data[sec]
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            SolverState(input_data, config_data)

        input_data, config_data = _create_valid_state_inputs()
        input_data[sec] = None
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            SolverState(input_data, config_data)


def test_solver_state_missing_subsection_keys():
    """Verifies exception handling when inner keys in IC, sim_params, or constraints are missing."""
    input_data, config_data = _create_valid_state_inputs()
    input_data["initial_conditions"]["velocity"] = None
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        SolverState(input_data, config_data)

    input_data, config_data = _create_valid_state_inputs()
    input_data["simulation_parameters"]["time_step"] = None
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        SolverState(input_data, config_data)

    input_data, config_data = _create_valid_state_inputs()
    input_data["physical_constraints"]["min_velocity"] = None
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        SolverState(input_data, config_data)


def test_solver_state_invalid_initial_velocity():
    """Verifies exception handling when initial_conditions.velocity is malformed."""
    input_data, config_data = _create_valid_state_inputs()

    input_data["initial_conditions"]["velocity"] = 5.0
    with pytest.raises((ValueError, RuntimeError)):
        SolverState(input_data, config_data)

    input_data["initial_conditions"]["velocity"] = [1.0, 2.0]
    with pytest.raises((ValueError, RuntimeError)):
        SolverState(input_data, config_data)


# ============================================================================
# NARRATIVE SECTION 3: Field Constraint Enforcement & Divergence Detection
# ============================================================================
# Field values are monitored for finite numerical stability:
#     not np.isfinite(fields).all() ==> raise ArithmeticError
# Allowing unconstrained physical evolution beyond arbitrary bounds.
# ============================================================================


def test_enforce_physical_constraints_unconstrained_evolution():
    """Verifies that field values evolve freely without artificial clamping."""
    input_data, config_data = _create_valid_state_inputs()
    state = SolverState(input_data, config_data)

    state.u[0, 0, 0] = 100.0
    state.v[0, 0, 0] = -100.0
    state.p[0, 0, 0] = -500.0

    state.enforce_physical_constraints()

    assert abs(state.u[0, 0, 0] - 100.0) < 1e-9
    assert abs(state.v[0, 0, 0] - (-100.0)) < 1e-9
    assert abs(state.p[0, 0, 0] - (-500.0)) < 1e-9


def test_enforce_physical_constraints_nan_inf_detection():
    """Verifies ArithmeticError detection when fields contain non-finite numbers (NaN or Inf)."""
    input_data, config_data = _create_valid_state_inputs()
    state = SolverState(input_data, config_data)

    state.fields[0, 0, 0, 0] = np.nan
    with pytest.raises(ArithmeticError, match="Numerical instability detected"):
        state.enforce_physical_constraints()

    state.fields[0, 0, 0, 0] = np.inf
    with pytest.raises(ArithmeticError, match="Numerical instability detected"):
        state.enforce_physical_constraints()


# ============================================================================
# NARRATIVE SECTION 4: Boundary Condition Dictionary Standard Extraction
# ============================================================================
# Boundary condition objects are normalized into pure dictionary representations,
# supporting both raw dicts and compiled C++ BoundaryCondition instances.
# ============================================================================


def test_get_boundary_condition_dicts_handling():
    """Verifies dictionary extraction across raw dicts, C++ objects, and fallback defaults."""
    input_data, config_data = _create_valid_state_inputs()

    class MockCppBC:
        def __init__(self):
            self.location = "y_max"
            self.type = "wall"
            self.u_val = 0.0
            self.v_val = 0.0
            self.w_val = 0.0
            self.scalar_p = 100000.0

    class MinimalObject:
        pass

    raw_dict = {"location": "x_min", "type": "inflow", "values": {"u": 10.0}}
    cpp_obj = MockCppBC()
    empty_obj = MinimalObject()

    input_data["boundary_conditions"] = [raw_dict, cpp_obj, empty_obj]
    state = SolverState(input_data, config_data)

    bc_dicts = state.get_boundary_condition_dicts()

    assert len(bc_dicts) == 3
    assert bc_dicts[0] == raw_dict
    assert bc_dicts[1]["location"] == "y_max"
    assert bc_dicts[1]["type"] == "wall"
    assert abs(bc_dicts[1]["values"]["p"] - 100000.0) < 1e-9
    assert bc_dicts[2]["location"] == ""
    assert bc_dicts[2]["type"] == ""
    assert bc_dicts[2]["values"]["u"] == 0.0
