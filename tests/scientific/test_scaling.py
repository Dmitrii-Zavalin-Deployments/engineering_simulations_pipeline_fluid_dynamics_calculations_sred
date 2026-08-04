# tests/scientific/test_scaling.py

import logging

import pytest

from src.step3.ops.scaling import get_dt_over_rho, get_rho_over_dt
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def setup_scaling_block(block, dt, rho):
    """
    Manually injects physical constants into the StencilBlock slots.
    Bypasses immutable constraints for targeted numerical testing.
    """
    object.__setattr__(block, '_dt', float(dt))
    object.__setattr__(block, '_rho', float(rho))
    return block


# --- ACCURACY GATE: NUMERICAL TRUTH ---


@pytest.mark.parametrize("dt, rho, expected_dt_rho, expected_rho_dt", [
    (0.01, 1.0, 0.01, 100.0),      # Standard Water-like case
    (0.001, 1000.0, 1e-6, 1e6),   # High density / Small step
    (0.5, 0.5, 1.0, 1.0),         # Unit symmetry
    (1.0, 1.225, 1/1.225, 1.225)  # Air-like density (STP)
])
def test_scaling_factors_accuracy(dt, rho, expected_dt_rho, expected_rho_dt):
    """Verifies scaling factors against analytical expectations."""
    block = make_step3_output_dummy()
    setup_scaling_block(block, dt, rho)

    assert get_dt_over_rho(block) == pytest.approx(expected_dt_rho, abs=1e-15)
    assert get_rho_over_dt(block) == pytest.approx(expected_rho_dt, abs=1e-15)


# --- SAFETY GATE: FORENSIC LOGGING & EXCEPTIONS ---


def test_scaling_rho_guard_logger(caplog):
    """Verify CRITICAL log and ValueError on vacuum/negative density."""
    block = make_step3_output_dummy()
    setup_scaling_block(block, dt=0.01, rho=0.0)  # Critical Physics Failure

    with caplog.at_level(logging.CRITICAL), pytest.raises(ValueError, match="Invalid rho"):
        get_dt_over_rho(block)

    assert "PHYSICS CRASH" in caplog.text
    assert "invalid density" in caplog.text


def test_scaling_dt_guard_logger(caplog):
    """Verify CRITICAL log and ZeroDivisionError on frozen time."""
    block = make_step3_output_dummy()
    setup_scaling_block(block, dt=0.0, rho=1.0)  # Critical Temporal Failure

    with caplog.at_level(logging.CRITICAL), pytest.raises(ZeroDivisionError, match="Invalid dt"):
        get_rho_over_dt(block)

    assert "TEMPORAL CRASH" in caplog.text
    assert "invalid time-step" in caplog.text


def test_scaling_instability_logger_dt_rho(caplog):
    """Verify ERROR log when dt/rho explodes into non-finite values."""
    block = make_step3_output_dummy()
    # High dt and sub-atomic rho to force an explosion
    setup_scaling_block(block, dt=1e308, rho=1e-308)

    with caplog.at_level(logging.ERROR), pytest.raises(ArithmeticError, match="dt/rho is non-finite"):
        get_dt_over_rho(block)

    assert "MATH FAILURE" in caplog.text


def test_scaling_instability_logger_rho_dt(caplog):
    """Verify ERROR log when rho/dt explodes into non-finite values."""
    block = make_step3_output_dummy()
    setup_scaling_block(block, dt=1e-308, rho=1e308)

    with caplog.at_level(logging.ERROR), pytest.raises(ArithmeticError, match="rho/dt is non-finite"):
        get_rho_over_dt(block)

    assert "MATH FAILURE" in caplog.text
