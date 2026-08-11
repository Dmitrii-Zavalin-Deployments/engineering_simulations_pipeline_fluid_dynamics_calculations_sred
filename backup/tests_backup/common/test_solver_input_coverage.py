# tests/common/test_solver_input_coverage.py

import pytest

from src.common.solver_input import DomainConfigInput, MaskInput


def test_solver_input_error_branches():
    # 1. Hit Line 51: Invalid Domain Type
    domain = DomainConfigInput()
    with pytest.raises(ValueError, match="Invalid domain type"):
        domain.type = "INVALID_TYPE"

    # 2. Hit Line 59: Invalid reference_velocity length
    with pytest.raises(ValueError, match="reference_velocity must have 3 items"):
        domain.reference_velocity = [1.0, 0.0]  # Only 2 items

    # 3. Hit Line 234: Invalid Mask values
    mask = MaskInput()
    with pytest.raises(ValueError, match="Mask contains invalid values"):
        mask.data = [0, 1, 99]  # 99 is not allowed

    print("✅ Logic-error branches in solver_input.py exercised.")