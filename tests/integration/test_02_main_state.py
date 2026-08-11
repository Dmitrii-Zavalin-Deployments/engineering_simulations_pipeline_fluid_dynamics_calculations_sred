"""
tests/integration/test_02_main_state.py
Integration Test 2: Main CLI -> Ingestion -> Sovereign SolverState initialization.
"""

import sys
from unittest.mock import patch

from src.state import SolverState


def test_main_cli_state_stage(workspace_folder, monkeypatch):
    """
    Narrative: Verifies the sovereign integration pipeline from command-line parsing
    through strict ingestion to sovereign SolverState initialization.
    """
    
    # We retrieve the temporary workspace folder path and input file handle 
    # for the current simulation run.
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]

    # We configure the simulated command-line arguments:
    #     python main.py --input_output_folder <folder> --input_file_name <input> --output_file_name <output>
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", "simulation_results.zip",
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # We establish a telemetry capture ledger for the instantiated sovereign SolverState object.
    captured_state = []

    def state_capture_wrapper(input_data, config_data):
        # The sovereign SolverState orchestrates physical simulation matrices from ingested data:
        #     instance = SolverState(input_data, config_data)
        instance = SolverState(input_data, config_data)
        captured_state.append(instance)
        # We short-circuit execution right after state creation:
        #     SystemExit(0)
        raise SystemExit(0)

    # We patch SolverState to intercept state initialization during main execution.
    with patch("src.main.SolverState", side_effect=state_capture_wrapper):
        try:
            from src.main import main
            main()
        except SystemExit as e:
            # The expected exit code for the intercepted sovereign state stage is:
            #     e.code == 0
            assert e.code == 0

    # We verify that exactly one state instance was captured, and inspect its structural properties.
    assert len(captured_state) == 1
    state = captured_state[0]
    
    # For the multi-dimensional simulation grid and field tensors, the expected shape tensor is:
    #     fields.shape = (4, 4, 4, 4)
    # For initial iteration tracking, the starting count is:
    #     total_iterations = 3
    assert state.fields.shape == (4, 4, 4, 4)
    assert state.total_iterations == 3
