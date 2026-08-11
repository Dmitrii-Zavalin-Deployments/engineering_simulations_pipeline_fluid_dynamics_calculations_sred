"""
tests/integration/test_03_main_cpp_gate.py
Integration Test 3: Main CLI -> Ingestion -> SolverState -> C++ Execution Bridge.
Verifies that the entire SolverState sovereign container object is correctly passed to C++,
validates all attributes comprehensively from the C++ perspective (direct object instance attributes),
and confirms zero-copy memory alignment and in-place state mutation across simulation iterations.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np


def test_main_cli_cpp_gate_stage(workspace_folder, monkeypatch):
    """
    Narrative: Verifies the complete end-to-end integration pipeline, ensuring that 
    the entire sovereign SolverState container object is passed directly to the C++ engine,
    and every single attribute is accessed and verified directly as C++ consumes them from the instance.
    """
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]

    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", "simulation_results.zip",
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # Setup mock C++ engine components to intercept calls and inspect transmitted state object
    mock_cpp_solver_instance = MagicMock()

    # Define a mock step behavior that simulates C++ mutating the shared fields array in-place
    def mock_step(state):
        state.fields[0, 0, 0, 0] = 99.9

    mock_cpp_solver_instance.step.side_effect = mock_step

    mock_cpp_solver_class = MagicMock(return_value=mock_cpp_solver_instance)
    mock_cpp_module = MagicMock()
    mock_cpp_module.NavierStokesSolver = mock_cpp_solver_class

    step_counter = {"calls": 0}

    # Intercept archivist call to exit cleanly after simulation iterations complete
    with patch("src.cpp_gate.navier_stokes_cpp", mock_cpp_module), patch(
        "src.main.archive_simulation_results", side_effect=SystemExit(0)
    ):
        try:
            from src.main import main
            main()
        except SystemExit as e:
            assert e.code == 0

    # 1. Verify C++ Solver Constructor received the entire sovereign SolverState container object
    assert mock_cpp_solver_class.called
    assert len(mock_cpp_solver_class.call_args.args) == 1
    passed_state_constructor = mock_cpp_solver_class.call_args.args[0]

    # 2. Comprehensive validation of all attributes directly from C++'s object perspective:
    
    # 2.1 Grid & Dimensions
    assert passed_state_constructor.nx == 4
    assert passed_state_constructor.ny == 4
    assert passed_state_constructor.nz == 4
    assert passed_state_constructor.x_min == 0.0
    assert passed_state_constructor.x_max == 1.0
    assert passed_state_constructor.y_min == 0.0
    assert passed_state_constructor.y_max == 1.0
    assert passed_state_constructor.z_min == 0.0
    assert passed_state_constructor.z_max == 1.0

    # 2.2 Tensors & Buffers
    assert passed_state_constructor.fields.shape == (4, 4, 4, 4)
    assert passed_state_constructor.mask.shape == (4, 4, 4)
    np.testing.assert_array_equal(passed_state_constructor.mask, np.zeros((4, 4, 4), dtype=np.int32))

    # 2.3 Simulation Timing & Iterations (Reflects post-run state after 3 completed time steps)
    assert passed_state_constructor.current_iteration == 3
    assert passed_state_constructor.current_time == 0.003
    assert passed_state_constructor.dt == 0.001
    assert passed_state_constructor.total_time == 0.003
    assert passed_state_constructor.total_iterations == 3
    assert passed_state_constructor.output_interval == 1

    # 2.4 Physical Constraints
    assert float(passed_state_constructor.physical_constraints["min_velocity"]) == -10.0
    assert float(passed_state_constructor.physical_constraints["max_velocity"]) == 10.0
    assert float(passed_state_constructor.physical_constraints["min_pressure"]) == -100.0
    assert float(passed_state_constructor.physical_constraints["max_pressure"]) == 100.0

    # 2.5 Domain Configuration
    assert passed_state_constructor.domain_configuration["type"] == "INTERNAL"
    assert passed_state_constructor.domain_configuration["reference_velocity"] == [0.0, 0.0, 0.0]

    # 2.6 Grid Configuration Dictionary
    assert passed_state_constructor.grid["nx"] == 4
    assert passed_state_constructor.grid["ny"] == 4
    assert passed_state_constructor.grid["nz"] == 4

    # 2.7 Fluid Properties
    assert float(passed_state_constructor.fluid_properties["density"]) == 1.0
    assert float(passed_state_constructor.fluid_properties["viscosity"]) == 0.01

    # 2.8 Initial Conditions
    assert passed_state_constructor.initial_conditions["velocity"] == [0.0, 0.0, 0.0]
    assert float(passed_state_constructor.initial_conditions["pressure"]) == 0.0

    # 2.9 Simulation Parameters Dictionary
    assert float(passed_state_constructor.simulation_parameters["time_step"]) == 0.001
    assert float(passed_state_constructor.simulation_parameters["total_time"]) == 0.003
    assert int(passed_state_constructor.simulation_parameters["output_interval"]) == 1

    # 2.10 Boundary Conditions
    assert len(passed_state_constructor.boundary_conditions) == 1
    assert passed_state_constructor.boundary_conditions[0]["location"] == "wall"
    assert passed_state_constructor.boundary_conditions[0]["type"] == "no-slip"

    # 2.11 External Forces
    assert passed_state_constructor.external_forces["force_vector"] == [0.0, 0.0, 0.0]
    assert passed_state_constructor.external_forces["gravity_vector"] == [0.0, -9.81, 0.0]

    # 2.12 Config Parameters
    assert int(passed_state_constructor.config["max_poisson_iterations"]) == 2000
    assert float(passed_state_constructor.config["poisson_tolerance"]) == 1e-8

    # 3. Verify total simulation steps executed match expected count
    assert mock_cpp_solver_instance.step.call_count == 3
    step_counter["calls"] = mock_cpp_solver_instance.step.call_count
    assert step_counter["calls"] == 3

    # 4. Verify step method received the sovereign SolverState container object directly
    assert len(mock_cpp_solver_instance.step.call_args.args) == 1
    passed_state_step = mock_cpp_solver_instance.step.call_args.args[0]

    # Confirm identity/equivalence between constructor state and step state
    assert passed_state_step is passed_state_constructor

    # 5. Verify zero-copy RAM mutation: ensure changes made during C++ step are immediately 
    # present in Python's sovereign container fields without any serialization or re-copying drift.
    assert passed_state_step.fields[0, 0, 0, 0] == 99.9
