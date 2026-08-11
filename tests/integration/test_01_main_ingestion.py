"""
tests/integration/test_01_main_ingestion.py
Integration Test 1: Main CLI entry point -> Ingestion schema parsing.
"""

import sys
from unittest.mock import patch

from src.ingestion import load_and_validate_inputs


def test_main_cli_ingestion_stage(workspace_folder, monkeypatch):
    """
    Narrative: Verifies the end-to-end integration of the Command Line Interface (CLI) 
    entry point with the strict ingestion and validation pipeline.
    """
    
    # We retrieve the temporary workspace folder and explicit file handles established 
    # for the current simulation run.
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    workspace_folder["config_path"]

    # We construct the simulated CLI argument vector mimicking user invocation:
    #     python main.py --input_output_folder <folder> --input_file_name <input> --output_file_name <output>
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", "simulation_results.zip",
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # We initialize a telemetry capture ledger to intercept the parsed data payloads 
    # returned by the ingestion module during execution.
    captured_return = []

    def spy_wrapper(*args, **kwargs):
        # The ingestion pipeline processes raw files into deterministic data dictionaries:
        #     res = (input_data, config_data)
        res = load_and_validate_inputs(*args, **kwargs)
        captured_return.append(res)
        return res

    # We deploy a mock spy wrapper around load_and_validate_inputs while short-circuiting 
    # downstream solver execution via SolverState to isolate the ingestion stage.
    with patch("src.main.load_and_validate_inputs", side_effect=spy_wrapper) as spy_ingest:
        with patch("src.main.SolverState") as mock_state_cls:
            # Execution terminates cleanly after state instantiation with exit code 0:
            #     SystemExit(0)
            mock_state_cls.side_effect = SystemExit(0)
            try:
                from src.main import main
                main()
            except SystemExit as e:
                # The expected exit code for a successful early-stage interception is:
                #     e.code == 0
                assert e.code == 0

        # We verify that the ingestion spy was successfully invoked exactly once,
        # and that the structural parameters match expected non-default schemas.
        assert spy_ingest.called
        assert len(captured_return) == 1
        
        # For grid resolution nx, the expected discretization count is:
        #     grid["nx"] = 4
        # For configuration Poisson iterations, the maximum limit is:
        #     max_poisson_iterations = 2000
        input_data, config_data = captured_return[0]
        assert input_data["grid"]["nx"] == 4
        assert config_data["max_poisson_iterations"] == 2000
