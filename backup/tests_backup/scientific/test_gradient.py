# tests/scientific/test_gradient.py

import logging

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step3.ops.gradient import compute_local_gradient_p
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def setup_analytical_gradient(p_func):
    """
    Wires a StencilBlock with a 1.0 unit coordinate system.
    Compliance: Rule 5 (Deterministic Defaults) and Rule 9 (Foundation Integrity).
    """
    # make_step3 returns a fully wired StencilBlock
    block = make_step3_output_dummy(nx=4, ny=4, nz=4)

    # Force unit spacing for clean analytical verification
    object.__setattr__(block, '_dx', 1.0)
    object.__setattr__(block, '_dy', 1.0)
    object.__setattr__(block, '_dz', 1.0)

    # 3D layout centered around (1,1,1) for finite difference mapping
    layout = {
        block.i_plus: (2, 1, 1), block.i_minus: (0, 1, 1),
        block.j_plus: (1, 2, 1), block.j_minus: (1, 0, 1),
        block.k_plus: (1, 1, 2), block.k_minus: (1, 1, 0)
    }

    for cell, (i, j, k) in layout.items():
        cell.set_field(FI.P, float(p_func(i, j, k)))

    return block


# --- PHYSICS ACCURACY GATE ---

def test_gradient_null_field():
    """Scenario 1: Hydrostatic Equilibrium. p=constant -> grad=(0,0,0)."""
    block = setup_analytical_gradient(lambda i, j, k: 10.0)

    grad = compute_local_gradient_p(block)

    assert grad == (0.0, 0.0, 0.0)


def test_gradient_linear_ascent():
    """Scenario 2: Unit Linear Gradient. p = x+y+z -> grad=(1,1,1)."""
    block = setup_analytical_gradient(lambda i, j, k: i + j + k)

    grad = compute_local_gradient_p(block)

    # Calculation: (p_ip - p_im) / (2 * dx) -> (2 - 0) / 2.0 = 1.0
    assert grad[0] == pytest.approx(1.0, abs=1e-15)
    assert grad[1] == pytest.approx(1.0, abs=1e-15)
    assert grad[2] == pytest.approx(1.0, abs=1e-15)


def test_gradient_alt_field_id():
    """Scenario 3: Verify operator targets P_NEXT when requested (Rule 9)."""
    block = make_step3_output_dummy()
    object.__setattr__(block, '_dx', 1.0)

    # Set gradient on P_NEXT, leaving P at zero
    block.i_plus.set_field(FI.P_NEXT, 5.0)
    block.i_minus.set_field(FI.P_NEXT, 1.0)

    grad = compute_local_gradient_p(block, field_id=FI.P_NEXT)

    assert grad[0] == 2.0  # (5-1)/2


# --- FORENSIC LOGGING & SAFETY GATE ---

def test_gradient_topology_crash_logger(caplog):
    """Verify CRITICAL log on missing neighbors (Rule 8)."""
    block = make_step3_output_dummy()
    # Break the stencil
    object.__setattr__(block, '_i_plus', None)

    with caplog.at_level(logging.CRITICAL), pytest.raises(AttributeError, match="Incomplete stencil"):
        compute_local_gradient_p(block)

    assert "TOPOLOGY CRASH" in caplog.text
    assert "missing neighbors" in caplog.text


def test_gradient_geometry_crash_logger(caplog):
    """Verify CRITICAL log on invalid grid spacing (Rule 7)."""
    block = make_step3_output_dummy()
    object.__setattr__(block, '_dx', 0.0)

    with caplog.at_level(logging.CRITICAL), pytest.raises(ZeroDivisionError, match="Invalid grid spacing"):
        compute_local_gradient_p(block)

    assert "GEOMETRY CRASH" in caplog.text


def test_gradient_numerical_instability_logger(caplog):
    """Verify ERROR log on non-finite results (Rule 7)."""
    block = setup_analytical_gradient(lambda i, j, k: 0.0)
    # Inject infinity to force explosion
    block.i_plus.set_field(FI.P, np.inf)

    with caplog.at_level(logging.ERROR), pytest.raises(ArithmeticError, match="gradient is non-finite"):
        compute_local_gradient_p(block)

    assert "MATH FAILURE" in caplog.text
    assert "Field: P" in caplog.text
