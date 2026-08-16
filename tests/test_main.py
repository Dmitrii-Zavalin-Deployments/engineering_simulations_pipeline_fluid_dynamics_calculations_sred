"""
@file test_main.py
@brief Literate Test Suite for Main Execution Control Plane (src/main.py)

This test module acts as a narrative specification and exhaustive verification suite for src/main.py.
Explanatory prose and physical time-integration formulas are written as commented prose, while
executable Python assertions verify argument validation, file presence checks, master simulation loop
orchestration, archivist double-fault recovery, and CLI control plane exit codes.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.main import main, run_simulation


def _write_json(path: Path, data: dict) -> Path:
    """Helper utility to serialize JSON dictionaries to disk."""
    with open(path, "w", encoding="utf-8") as f:
        import json
        json.dump(data, f)
    return path


# ============================================================================
# NARRATIVE SECTION 1: Argument Nullability & File Existence Verification
# ============================================================================
# Non-default execution policy dictates that all folder and file parameters
# must be non-null and point to valid, existing filesystem resources.
#
# Boundary checks enforce:
#     input_output_folder != None
#     input_file_name != None
#     output_file_name != None
#     is_file(input_output_folder / input_file_name) == True
#     is_file(BASE_DIR / "config" / "config.json") == True
# ============================================================================


def test_run_simulation_null_arguments():
    """Verifies that passing None to any parameter triggers a fatal ValueError."""
    # We test null input_output_folder:
    with pytest.raises(ValueError, match="input_output_folder must be explicitly provided"):
        run_simulation(None, "input.json", "output.json")

    # We test null input_file_name:
    with pytest.raises(ValueError, match="input_file_name must be explicitly provided"):
        run_simulation("/some/dir", None, "output.json")

    # We test null output_file_name:
    with pytest.raises(ValueError, match="output_file_name must be explicitly provided"):
        run_simulation("/some/dir", "input.json", None)


def test_run_simulation_missing_input_file(tmp_path):
    """Verifies FileNotFoundError when the input JSON configuration file is missing from disk."""
    missing_file_name = "non_existent_input.json"

    # Input path evaluated as out_path / input_file_name does not exist:
    with pytest.raises(FileNotFoundError, match="Input configuration file not found at"):
        run_simulation(tmp_path, missing_file_name, "output.json")


def test_run_simulation_missing_config_file(tmp_path):
    """Verifies FileNotFoundError when system config/config.json is absent relative to BASE_DIR."""
    # Write valid input file in workspace:
    input_file = _write_json(tmp_path / "navier_stokes_input.json", {"test": "data"})

    # Point BASE_DIR to empty directory lacking config/config.json:
    fake_base_dir = tmp_path / "fake_base"
    fake_base_dir.mkdir()

    with patch("src.main.BASE_DIR", fake_base_dir), pytest.raises(
        FileNotFoundError, match="Configuration file not found at"
    ):
        run_simulation(tmp_path, input_file.name, "output.json")


# ============================================================================
# NARRATIVE SECTION 2: Discrete Time-Integration Master Loop Integration
# ============================================================================
# The discrete time evolution of fluid state obeys:
#     t_{n+1} = t_n + dt
#     N_{steps} = total_iterations
#
# For each step n from 1 to N_{steps}:
#     1. step_simulation(state) -> C++ Navier-Stokes velocity/pressure step
#     2. state.enforce_physical_constraints() -> numerical stability clip
#
# Upon loop completion without error:
#     archive_simulation_results(state, status="SUCCESS")
# ============================================================================


def test_run_simulation_success_pipeline(tmp_path):
    """Verifies end-to-end master loop execution, constraint enforcement, and SUCCESS archiving."""
    # Setup mock workspace with valid input and config files:
    input_file = _write_json(tmp_path / "input.json", {"grid": {}})
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "config.json", {"max_poisson_iterations": 10})

    mock_state = MagicMock()
    mock_state.total_iterations = 2
    mock_state.dt = 0.001
    mock_state.current_iteration = 0
    mock_state.current_time = 0.0

    # Execute simulation under patched dependencies:
    with patch("src.main.BASE_DIR", tmp_path), \
         patch("src.main.load_and_validate_inputs", return_value=({}, {})), \
         patch("src.main.SolverState", return_value=mock_state), \
         patch("src.main.step_simulation") as mock_step, \
         patch("src.main.archive_simulation_results") as mock_archive:

        # Simulate iteration advancement inside loop:
        def step_side_effect(state):
            state.current_iteration += 1
            state.current_time += state.dt

        mock_step.side_effect = step_side_effect

        run_simulation(
            input_output_folder=tmp_path,
            input_file_name=input_file.name,
            output_file_name="output_manifest.json",
        )

        # Assert physical step and constraint methods were invoked N_steps times:
        assert mock_step.call_count == 2
        assert mock_state.enforce_physical_constraints.call_count == 2

        # Assert success manifest packaging was triggered:
        mock_archive.assert_called_once_with(
            state=mock_state,
            output_dir=tmp_path.resolve(),
            output_filename="output_manifest.json",
            status="SUCCESS",
        )


# ============================================================================
# NARRATIVE SECTION 3: Failure Mode Manifest Archiving & Double-Fault Recovery
# ============================================================================
# When master time integration fails (e.g., CFD instability or divergence):
#     exec_err caught during loop execution
#     --> archivist triggered with status="FAILURE"
#
# Double-Fault Protection:
# If archivist manifest generation ALSO encounters a system fault (OSError/ValueError):
#     critical log emitted
#     --> original exception re-raised uninterrupted
# ============================================================================


def test_run_simulation_loop_failure_writes_failure_manifest(tmp_path):
    """Verifies that an unhandled exception in step_simulation triggers failure status archiving."""
    input_file = _write_json(tmp_path / "input.json", {})
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "config.json", {})

    mock_state = MagicMock()
    mock_state.total_iterations = 5
    mock_state.dt = 0.001

    with patch("src.main.BASE_DIR", tmp_path), \
         patch("src.main.load_and_validate_inputs", return_value=({}, {})), \
         patch("src.main.SolverState", return_value=mock_state), \
         patch("src.main.step_simulation", side_effect=RuntimeError("Solver diverged")), \
         patch("src.main.archive_simulation_results") as mock_archive, \
         pytest.raises(RuntimeError, match="Solver diverged"):
        run_simulation(
            input_output_folder=tmp_path,
            input_file_name=input_file.name,
            output_file_name="output_manifest.json",
        )

    # Assert archivist was called to emit a FAILURE status manifest:
    mock_archive.assert_called_once_with(
        state=mock_state,
        output_dir=tmp_path.resolve(),
        output_filename="output_manifest.json",
        status="FAILURE",
    )


def test_run_simulation_double_fault_handling(tmp_path):
    """Verifies resilience when both simulation execution AND failure manifest creation fail."""
    input_file = _write_json(tmp_path / "input.json", {})
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "config.json", {})

    mock_state = MagicMock()
    mock_state.total_iterations = 1
    mock_state.dt = 0.001

    with patch("src.main.BASE_DIR", tmp_path), \
         patch("src.main.load_and_validate_inputs", return_value=({}, {})), \
         patch("src.main.SolverState", return_value=mock_state), \
         patch("src.main.step_simulation", side_effect=ValueError("Divergent field value")), \
         patch("src.main.archive_simulation_results", side_effect=OSError("Disk read-only")), \
         pytest.raises(ValueError, match="Divergent field value"):
        run_simulation(
            input_output_folder=tmp_path,
            input_file_name=input_file.name,
            output_file_name="output_manifest.json",
        )


# ============================================================================
# NARRATIVE SECTION 4: CLI Entrypoint & System Exit Code Contracts
# ============================================================================
# Main entrypoint parses CLI arguments:
#     --input_output_folder, --input_file_name, --output_file_name
#
# Unhandled operational exceptions mapped during main execution:
#     (FileNotFoundError, ValueError, KeyError, OSError, RuntimeError, TypeError, AttributeError)
#     --> Output FATAL PIPELINE ERROR to stderr
#     --> Exit process with sys.exit(1)
# ============================================================================


def test_main_cli_success(tmp_path):
    """Verifies CLI entrypoint parsing arguments and invoking run_simulation successfully."""
    test_args = [
        "main.py",
        "--input_output_folder", str(tmp_path),
        "--input_file_name", "input.json",
        "--output_file_name", "output.json",
    ]

    with patch.object(sys, "argv", test_args), \
         patch("src.main.run_simulation") as mock_run:

        main()
        mock_run.assert_called_once_with(
            input_output_folder=str(tmp_path),
            input_file_name="input.json",
            output_file_name="output.json",
        )


def test_main_cli_fatal_error_exit(capsys):
    """Verifies that fatal errors caught during main execution print to stderr and exit with code 1."""
    test_args = [
        "main.py",
        "--input_output_folder", "/invalid/dir",
        "--input_file_name", "input.json",
        "--output_file_name", "output.json",
    ]

    with patch.object(sys, "argv", test_args), \
         patch("src.main.run_simulation", side_effect=FileNotFoundError("Input path invalid")), \
         pytest.raises(SystemExit) as exc_info:
        main()

    # System exit code assertion:
    assert exc_info.value.code == 1

    # Error log written to stderr:
    captured = capsys.readouterr()
    assert "FATAL PIPELINE ERROR: Input path invalid" in captured.err
