"""
tests/integration/test_01_main_ingestion.py
Integration Test 1: Main CLI entry point -> Ingestion schema parsing.
"""

import sys
from unittest.mock import patch

from src.ingestion import load_and_validate_inputs


def test_main_cli_ingestion_stage(workspace_folder, monkeypatch):
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    workspace_folder["config_path"]

    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", "simulation_results.zip",
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # Spy on load_and_validate_inputs to verify ingestion is called with correct resolved paths
    with patch("src.main.load_and_validate_inputs", wraps=load_and_validate_inputs) as spy_ingest:
        with patch("src.main.SolverState") as mock_state_cls:
            # Short-circuit execution after state instantiation for stage 1 test
            mock_state_cls.side_effect = SystemExit(0)
            try:
                from src.main import main
                main()
            except SystemExit as e:
                assert e.code == 0

        assert spy_ingest.called
        input_data, config_data = spy_ingest.call_args.result
        assert input_data["grid"]["nx"] == 4
        assert config_data["max_poisson_iterations"] == 2000
