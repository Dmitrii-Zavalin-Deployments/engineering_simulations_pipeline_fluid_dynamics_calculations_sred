# tests/scientific/test_ppe_solver.py

import logging

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step3.ppe_solver import solve_pressure_poisson_step
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

logger = logging.getLogger(__name__)


def setup_ppe_block(dt=1.0, rho=1.0, dx=1.0):
    """
    Sets up a StencilBlock for PPE testing with Unit Geometry.
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10]  # Central core block

    # Bypass read-only properties to force Unit Geometry for clean math
    object.__setattr__(block, '_dt', float(dt))
    object.__setattr__(block, '_rho', float(rho))
    object.__setattr__(block, '_dx', float(dx))
    object.__setattr__(block, '_dy', float(dx))
    object.__setattr__(block, '_dz', float(dx))

    # Clean buffer for analytical transparency
    block.center.fields_buffer.fill(0.0)
    return block


# --- PHYSICS CONVERGENCE TESTS ---

def test_ppe_unit_convergence():
    """
    Verifies p_new with dx=1, P=0, omega=1 (Gauss-Seidel).
    Formula: p_new = (1/stencil_denom) * (sum_neighbors - rhs)
    Setup: div_v = 1.0, stencil_denom = 6.0, rhs = 1.0
    Expectation: (1/6) * (0 - 1.0) = -0.1666...
    """
    block = setup_ppe_block(dt=1.0, rho=1.0, dx=1.0)

    # Create div_v = 1.0 via VX_STAR gradient (2-0)/2
    block.i_plus.set_field(FI.VX_STAR, 2.0)
    block.i_minus.set_field(FI.VX_STAR, 0.0)

    # solve_pressure_poisson_step(block, threshold, omega)
    residual = solve_pressure_poisson_step(block, 1e6, 1.0)

    expected_p = -1.0 / 6.0
    actual_p = block.center.get_field(FI.P_NEXT)

    assert actual_p == pytest.approx(expected_p)
    assert residual == pytest.approx(abs(expected_p))


def test_ppe_rhie_chow_cancellation():
    """
    Checks if Rhie-Chow stabilization properly offsets the source.
    If div_v_star == dt * laplacian(P), the RHS must be 0.
    """
    block = setup_ppe_block(dt=1.0, rho=1.0, dx=1.0)

    # 1. Set div_v = 1.0
    block.i_plus.set_field(FI.VX_STAR, 2.0)
    block.i_minus.set_field(FI.VX_STAR, 0.0)

    # 2. Set P fields such that laplacian(P) = 1.0
    # lap = (1 - 0 + 0) / 1^2 = 1.0
    block.i_plus.set_field(FI.P, 1.0)
    block.center.set_field(FI.P, 0.0)
    block.i_minus.set_field(FI.P, 0.0)

    # RHS = (rho/dt) * (div_v - dt * lap_p) = 1 * (1 - 1*1) = 0
    solve_pressure_poisson_step(block, 1e6, 1.0)

    assert block.center.get_field(FI.P_NEXT) == 0.0


def test_ppe_sor_relaxation_logic():
    """Verifies omega scales the update: p_new = (1-w)p_old + (w/denom)(...)"""
    block = setup_ppe_block(dt=1.0, rho=1.0, dx=1.0)
    omega = 0.5

    # Start with p_old = 10.0
    block.center.set_field(FI.P_NEXT, 10.0)

    # With sum_neighbors=0 and rhs=0, p_new = (1-0.5)*10 + 0 = 5.0
    solve_pressure_poisson_step(block, 1e6, omega)

    assert block.center.get_field(FI.P_NEXT) == pytest.approx(5.0)


# --- FORENSIC LOGGING & SAFETY TESTS ---

def test_ppe_poisoned_input_audit(caplog):
    """Verifies Rule 7: Fail-Fast on poisoned (NaN) pressure trial."""
    block = setup_ppe_block()
    block.center.set_field(FI.P_NEXT, np.nan)

    with caplog.at_level(logging.ERROR, logger="Solver.PPE"), pytest.raises(
        ArithmeticError, match="Poisoned Pressure"
    ):
        solve_pressure_poisson_step(block, 1e6, 1.0)

    assert "PPE CRITICAL" in caplog.text


def test_ppe_dna_leak_correction_log(caplog):
    """Checks if the DNA AUDIT warning triggers for array promotion."""
    block = setup_ppe_block()

    # Force an array leak by setting a neighbor as a single-element array
    block.i_plus.set_field(FI.P_NEXT, np.array([1.0]))

    with caplog.at_level(logging.WARNING, logger="Solver.PPE"):
        solve_pressure_poisson_step(block, 1e6, 1.0)

    result = block.center.get_field(FI.P_NEXT)
    assert isinstance(result, float), "Rule 9 Violation: Field access must return a Sovereign Scalar."


def test_ppe_non_finite_divergence_guard(caplog):
    """Ensures divergence instability is caught before the solve."""
    block = setup_ppe_block()
    # Inject NaN into the star velocity field
    block.i_plus.set_field(FI.VX_STAR, np.nan)

    with caplog.at_level(logging.ERROR, logger="Solver.PPE"), pytest.raises(
        ArithmeticError, match="Divergence exploded"
    ):
        solve_pressure_poisson_step(block, 1e6, 1.0)

    assert "MATH FAILURE" in caplog.text
