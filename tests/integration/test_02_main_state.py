"""
tests/integration/test_02_main_state.py
Integration Test 2: Main CLI -> Ingestion -> Sovereign SolverState initialization.
"""

import sys
from unittest.mock import patch

from src.state import SolverState


def test_main_cli_state_stage(workspace_folder, monkeypatch):
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]

    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", "simulation_results.zip",
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    captured_state = []

    def state_capture_wrapper(input_data, config_data):
        instance = SolverState(input_data, config_data)
        captured_state.append(instance)
        raise SystemExit(0)  # Halt pipeline after state creation for stage 2

    with patch("src.main.SolverState", side_effect=state_capture_wrapper):
        try:
            from src.main import main
            main()
        except SystemExit as e:
            assert e.code == 0

    assert len(captured_state) == 1
    state = captured_state[0]
    assert state.fields.shape == (4, 4, 4, 4)
    assert state.total_iterations == 3
