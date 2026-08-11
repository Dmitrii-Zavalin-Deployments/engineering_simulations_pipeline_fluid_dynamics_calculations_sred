"""
tests/integration/test_03_main_cpp_gate.py
Integration Test 3: Main CLI -> Ingestion -> SolverState -> C++ Execution Bridge.
Verifies that all data and parameters from the sovereign container are correctly passed to C++.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np


def test_main_cli_cpp_gate_stage(workspace_folder, monkeypatch):
    """
    Narrative: Verifies the complete end-to-end integration pipeline, ensuring that 
    all fields, physical constraints, boundary conditions, fluid properties, and 
    production configuration parameters from the sovereign SolverState container 
    are accurately unpacked and passed through the C++ bridge to the compiled engine.
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

    # Setup mock C++ engine components to intercept calls and inspect transmitted data
    mock_cpp_solver_instance = MagicMock()
    mock_cpp_solver_class = MagicMock(return_value=mock_cpp_solver_instance)
    
    class MockBoundaryCondition:
        def __init__(self):
            self.location = None
            self.type = None
            self.u_val = 0.0
            self.v_val = 0.0
            self.w_val = 0.0
            self.scalar_p = 0.0

    mock_cpp_module = MagicMock()
    mock_cpp_module.NavierStokesSolver = mock_cpp_solver_class
    mock_cpp_module.BoundaryCondition = MockBoundaryCondition

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

    # 1. Verify C++ Solver Constructor received all required configuration and grid parameters
    assert mock_cpp_solver_class.called
    constructor_args = mock_cpp_solver_class.call_args.kwargs
    
    assert constructor_args["nx"] == 4
    assert constructor_args["ny"] == 4
    assert constructor_args["nz"] == 4
    assert constructor_args["max_poisson_iters"] == 2000  # Synced from production config/config.json
    assert constructor_args["poisson_tolerance"] == 1e-8    # Synced from production config/config.json
    assert constructor_args["density"] == 1.0

    # 2. Verify total simulation steps executed (total_time 0.003 / time_step 0.001 = 3 iterations)
    assert mock_cpp_solver_instance.step.call_count == 3
    step_counter["calls"] = mock_cpp_solver_instance.step.call_count
    assert step_counter["calls"] == 3

    # 3. Inspect step parameters passed to the C++ solver during execution
    step_args = mock_cpp_solver_instance.step.call_args[0]
    fields_arg, mask_arg, fx_arg, fy_arg, fz_arg, bc_list_arg, dt_arg, mu_arg = step_args

    # Verify tensor arrays and memory alignment
    assert isinstance(fields_arg, np.ndarray)
    assert fields_arg.shape == (4, 4, 4, 4)
    assert isinstance(mask_arg, np.ndarray)
    assert mask_arg.shape == (4, 4, 4)

    # Verify external force fields including gravity integration (-9.81 in y)
    np.testing.assert_array_equal(fx_arg, np.zeros((4, 4, 4)))
    np.testing.assert_array_equal(fy_arg, np.full((4, 4, 4), -9.81))
    np.testing.assert_array_equal(fz_arg, np.zeros((4, 4, 4)))

    # Verify boundary conditions mapping (including velocity and pressure components)
    assert len(bc_list_arg) == 1
    bc_mapped = bc_list_arg[0]
    assert bc_mapped.location == "wall"
    assert bc_mapped.type == "no-slip"
    assert bc_mapped.u_val == 0.0
    assert bc_mapped.v_val == 0.0
    assert bc_mapped.w_val == 0.0
    assert bc_mapped.scalar_p == 0.0

    # Verify time step and viscosity values
    assert dt_arg == 0.001
    assert mu_arg == 0.01
