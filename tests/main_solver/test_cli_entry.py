# tests/main_solver/test_cli_entry.py

import sys
from unittest.mock import patch

import pytest
from src.main import main


def test_main_entry_point_missing_args():
    """Validates Line 156-158: Proper exit when no args provided."""
    with patch.object(sys, 'argv', ['src/main_solver.py']), pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1


def test_main_entry_point_success():
    """Validates Line 168-170: Execution through the __main__ gateway."""
    # We mock run_solver to avoid running the whole physics engine
    with (
        patch("src.main.run_solver", return_value="test_artifacts.zip"),
        patch.object(sys, 'argv', ['src/main_solver.py', 'dummy.json']),
        pytest.raises(SystemExit) as e,
    ):
        main()
    assert e.value.code == 0


def test_main_entry_point_fatal_error():
    """Validates Line 163-166: Catching global exceptions in main."""
    with (
        patch("src.main.run_solver", side_effect=RuntimeError("System Collapse")),
        patch.object(sys, 'argv', ['src/main_solver.py', 'dummy.json']),
        pytest.raises(SystemExit) as e,
    ):
        main()
    assert e.value.code == 1
