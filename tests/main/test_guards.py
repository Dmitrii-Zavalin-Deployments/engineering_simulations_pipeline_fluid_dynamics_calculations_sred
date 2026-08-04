# tests/main/test_guards.py

import sys
from unittest.mock import patch

import pytest

# Core imports from the solver
from src.main import _load_simulation_context, main

# --- 1. File System Presence Guards ---

def test_load_context_missing_input_file():
    """Validates the solver terminates if the primary input JSON is not found."""
    with patch("src.main.Path.exists") as mock_exists:
        # First check (input_path) returns False
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError, match="Input file missing"):
            _load_simulation_context("non_existent_input.json")

def test_load_context_missing_config_file():
    """Validates the solver terminates if the required config.json is missing."""
    with patch("src.main.Path.exists") as mock_exists:
        # First check (input) is True, Second check (config) is False
        mock_exists.side_effect = [True, False]
        with pytest.raises(FileNotFoundError, match="config.json required"):
            _load_simulation_context("valid_input.json")

# --- 2. CLI Entry Point & Orchestration Guards ---

def test_cli_entrypoint_no_args():
    """
    Validates the Usage prompt and Exit(1) when no path is provided.
    Uses the direct main() call to avoid runpy double-import warnings.
    """
    with patch("sys.argv", ["src/main.py"]), \
         patch("builtins.print") as mock_print:
        
        with pytest.raises(SystemExit) as e:
            main()
            
        assert e.value.code == 1
        mock_print.assert_any_call("Usage: python src/main.py <input_json_path>")

def test_cli_entrypoint_success():
    """
    Validates a successful end-to-end orchestration.
    Ensures SystemExit(0) is raised and the correct archive path is printed.
    """
    # Patch the orchestrator to simulate a successful physics run
    with patch("src.main.run_solver", return_value="mock.zip"), \
         patch("sys.argv", ["src/main.py", "valid.json"]), \
         patch("builtins.print") as mock_print:
        
        with pytest.raises(SystemExit) as e:
            main()
            
        # Verify the success contract
        assert e.value.code == 0
        mock_print.assert_any_call("Pipeline complete. Artifacts archived at: mock.zip")

def test_cli_entrypoint_fatal_error():
    """
    Validates the FATAL error handling path (Exit 1) for unhandled exceptions.
    Ensures that errors are reported to stderr as per Rule 8.
    """
    with patch("src.main.run_solver", side_effect=RuntimeError("System Collapse")), \
         patch("sys.argv", ["src/main.py", "valid.json"]), \
         patch("builtins.print") as mock_print, \
         patch("traceback.print_exc"): # Silent traceback for cleaner test logs
        
        with pytest.raises(SystemExit) as e:
            main()
            
        assert e.value.code == 1
        mock_print.assert_any_call("FATAL PIPELINE ERROR: System Collapse", file=sys.stderr)
