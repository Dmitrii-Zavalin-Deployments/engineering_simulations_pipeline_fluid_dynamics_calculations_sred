"""
tests/integration/test_03_main_cpp_gate.py
Integration Test 3: Main CLI -> Ingestion -> SolverState -> C++ Execution Bridge.
"""

import sys
from unittest.mock import patch

from src.cpp_gate import step_simulation


def test_main_cli_cpp_gate_stage(workspace_folder, monkeypatch):
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]

    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    step_counter = {"calls": 0}

    def step_wrapper(state):
        step_simulation(state)
        step_counter["calls"] += 1

    # Intercept archivist call to exit before file archiving
    with patch("src.main.step_simulation", side_effect=step_wrapper):
        with patch("src.main.archive_simulation_results", side_effect=SystemExit(0)):
            try:
                from src.main import main
                main()
            except SystemExit as e:
                assert e.code == 0

    # Total iterations defined in conftest input data is 3
    assert step_counter["calls"] == 3
