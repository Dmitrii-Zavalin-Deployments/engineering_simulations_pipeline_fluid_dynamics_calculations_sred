# tests/main_solver/test_contracts.py

from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from src.main import run_solver
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


def test_run_solver_input_schema_violation():
    with patch("src.main._load_simulation_context") as mock_load, \
         patch("src.main.open", create=True) as mock_open:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_open.return_value.__enter__.return_value.read.return_value = "{}"
        with patch("jsonschema.validate") as mock_validate:
            mock_validate.side_effect = jsonschema.exceptions.ValidationError("Invalid Input")
            with pytest.raises(jsonschema.exceptions.ValidationError, match="Invalid Input"):
                run_solver("dummy.json")

def test_run_solver_state_schema_violation(caplog):
    valid_input = create_validated_input()
    real_state = make_step4_output_dummy() 
    with patch("src.main._load_simulation_context") as mock_load:
        mock_load.return_value = MagicMock(input_data=valid_input)
        with patch("src.common.solver_state.SolverState.validate_against_schema") as mock_val:
            mock_val.side_effect = jsonschema.exceptions.ValidationError("State Mismatch")
            with patch("src.main.orchestrate_step1", return_value=real_state), \
                 patch("src.main.orchestrate_step2", return_value=real_state):
                with pytest.raises(jsonschema.exceptions.ValidationError, match="State Mismatch"):
                    run_solver("dummy.json")
                assert "!!! STATE CONTRACT VIOLATION" in caplog.text