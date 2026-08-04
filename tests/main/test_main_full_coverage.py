# tests/main/test_main_full_coverage.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.main import main, run_solver
from tests.helpers.solver_input_schema_dummy import (
    create_validated_input,
    get_explicit_solver_config,
)


def _load_base_config() -> dict:
    """Loads production configuration parameters from config/config.json."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_run_solver_with_output_path(tmp_path):
    """Cover line 139: run_solver with valid output_path."""
    input_file = tmp_path / "input.json"
    config = get_explicit_solver_config(2, 2, 2)
    config["simulation_parameters"]["total_time"] = 0.0
    config["simulation_parameters"].update(_load_base_config())
    input_file.write_text(json.dumps(config))
    
    with patch("src.main._load_simulation_context") as mock_load, \
         patch("src.main._configure_numerical_runtime"), \
         patch("src.main.orchestrate_step1") as mock_step1, \
         patch("src.main.orchestrate_step2") as mock_step2, \
         patch("src.main.ElasticManager"), \
         patch("src.main.archive_simulation_artifacts") as mock_archive:
        
        mock_context = MagicMock()
        mock_context.input_data = create_validated_input(2, 2, 2)
        mock_load.return_value = mock_context
        
        mock_state = MagicMock()
        mock_state.ready_for_time_loop = False
        mock_step1.return_value = mock_state
        mock_step2.return_value = mock_state
        
        mock_archive.return_value = "/tmp/archived.zip"
        
        out_path = tmp_path / "output.zip"
        res = run_solver(input_file, output_path=out_path)
        assert res == "/tmp/archived.zip"
        mock_archive.assert_called_once_with(mock_state, output_path=str(out_path))


def test_run_solver_output_path_typeerror_fallback(tmp_path):
    """Cover lines 141-142: TypeError fallback in archive_simulation_artifacts with output_path."""
    input_file = tmp_path / "input.json"
    config = get_explicit_solver_config(2, 2, 2)
    config["simulation_parameters"]["total_time"] = 0.0
    config["simulation_parameters"].update(_load_base_config())
    input_file.write_text(json.dumps(config))
    
    with patch("src.main._load_simulation_context") as mock_load, \
         patch("src.main._configure_numerical_runtime"), \
         patch("src.main.orchestrate_step1") as mock_step1, \
         patch("src.main.orchestrate_step2") as mock_step2, \
         patch("src.main.ElasticManager"), \
         patch("src.main.archive_simulation_artifacts") as mock_archive:
        
        mock_context = MagicMock()
        mock_context.input_data = create_validated_input(2, 2, 2)
        mock_load.return_value = mock_context
        
        mock_state = MagicMock()
        mock_state.ready_for_time_loop = False
        mock_step1.return_value = mock_state
        mock_step2.return_value = mock_state
        
        def side_effect(state, output_path=None):
            if output_path is not None:
                raise TypeError("Invalid argument")
            return "/tmp/fallback.zip"
            
        mock_archive.side_effect = side_effect
        
        out_path = tmp_path / "output.zip"
        res = run_solver(input_file, output_path=out_path)
        assert res == "/tmp/fallback.zip"
        assert mock_archive.call_count == 2


def test_main_cli_input_output_folder_and_file(monkeypatch, tmp_path):
    """Cover line 177 and line 187: --input_output_folder with --input_file_name and --output_file_name."""
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    input_file = input_dir / "navier_stokes_solver_input.json"
    config = get_explicit_solver_config(2, 2, 2)
    config["simulation_parameters"].update(_load_base_config())
    input_file.write_text(json.dumps(config))
    
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--input_output_folder",
            str(input_dir),
            "--input_file_name",
            "navier_stokes_solver_input.json",
            "--output_file_name",
            "out.zip",
        ],
    )
    
    with patch("src.main.run_solver") as mock_run:
        mock_run.return_value = "/tmp/out.zip"
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once()


def test_main_cli_output_file_name_only(monkeypatch, tmp_path):
    """Cover line 189: --output_file_name without --input_output_folder (using positional input)."""
    input_file = tmp_path / "input.json"
    config = get_explicit_solver_config(2, 2, 2)
    config["simulation_parameters"].update(_load_base_config())
    input_file.write_text(json.dumps(config))
    
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            str(input_file),
            "--output_file_name",
            "standalone_out.zip",
        ],
    )
    
    with patch("src.main.run_solver") as mock_run:
        mock_run.return_value = "/tmp/standalone_out.zip"
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once()
