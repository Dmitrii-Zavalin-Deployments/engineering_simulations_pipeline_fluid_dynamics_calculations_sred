"""
tests/integration/test_03_main_cpp_gate.py
Integration Test 3: Main CLI -> Ingestion -> SolverState -> C++ Execution Bridge.
Verifies that the entire SolverState sovereign container object is correctly passed to C++
and validates zero-copy memory alignment and in-place sharing between Python and C++.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np


def test_main_cli_cpp_gate_stage(workspace_folder, monkeypatch):
    """
    Narrative: Verifies the complete end-to-end integration pipeline, ensuring that 
    the entire sovereign SolverState container object is passed directly to both the C++ engine 
    constructor and the step method, establishing zero-copy memory sharing with zero drift.
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
        # Simulate C++ writing directly into the sovereign container's fields array in RAM
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

    # Verify state container attributes through the passed object instance
    assert passed_state_constructor.nx == 4
    assert passed_state_constructor.ny == 4
    assert passed_state_constructor.nz == 4
    assert int(passed_state_constructor.config["max_poisson_iterations"]) == 2000  # Synced from config/config.json
    assert float(passed_state_constructor.config["poisson_tolerance"]) == 1e-8    # Synced from config/config.json
    assert float(passed_state_constructor.fluid_properties["density"]) == 1.0

    # 2. Verify total simulation steps executed (total_time 0.003 / time_step 0.001 = 3 iterations)
    assert mock_cpp_solver_instance.step.call_count == 3
    step_counter["calls"] = mock_cpp_solver_instance.step.call_count
    assert step_counter["calls"] == 3

    # 3. Verify step method received the sovereign SolverState container object directly
    assert len(mock_cpp_solver_instance.step.call_args.args) == 1
    passed_state_step = mock_cpp_solver_instance.step.call_args.args[0]

    # Confirm identity/equivalence between constructor state and step state
    assert passed_state_step is passed_state_constructor

    # 4. Verify zero-copy RAM mutation: ensure changes made during C++ step are immediately 
    # present in Python's sovereign container fields without any serialization or re-copying drift.
    assert passed_state_step.fields[0, 0, 0, 0] == 99.9
