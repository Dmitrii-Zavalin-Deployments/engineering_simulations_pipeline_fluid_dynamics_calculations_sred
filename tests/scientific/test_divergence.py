# tests/scientific/test_divergence.py

import logging
import math

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step3.ops.divergence import compute_local_divergence_v_star
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy

# Rule 7: Granular Traceability
logger = logging.getLogger(__name__)


def setup_divergent_stencil(u_func, v_func, w_func):
    """
    Rule 9 Bridge: Uses the production Step 3 Dummy.
    Sets up a unit coordinate system for analytical divergence verification.
    """
    # Get a real production block (wires up real Cell objects to the buffer)
    block = make_step3_output_dummy(nx=4, ny=4, nz=4)
    
    # Rule 5: Deterministic Spacing for math purity
    object.__setattr__(block, '_dx', 1.0)
    object.__setattr__(block, '_dy', 1.0)
    object.__setattr__(block, '_dz', 1.0)

    # Centralized layout for finite difference neighbors
    layout = {
        block.i_plus: (2, 1, 1), block.i_minus: (0, 1, 1),
        block.j_plus: (1, 2, 1), block.j_minus: (1, 0, 1),
        block.k_plus: (1, 1, 2), block.k_minus: (1, 1, 0)
    }
    
    for cell, (i, j, k) in layout.items():
        # Map functions to the STAR velocity fields (Target for Divergence)
        cell.set_field(FI.VX_STAR, float(u_func(i, j, k)))
        cell.set_field(FI.VY_STAR, float(v_func(i, j, k)))
        cell.set_field(FI.VZ_STAR, float(w_func(i, j, k)))
            
    return block


# --- PHYSICS VALIDATION SCENARIOS ---

def test_divergence_zero_field():
    """Scenario 1: Null Field (The Zero-Gate). v=(0,0,0) -> div=0."""
    block = setup_divergent_stencil(lambda i, j, k: 0, lambda i, j, k: 0, lambda i, j, k: 0)
    result = compute_local_divergence_v_star(block)
    assert result == 0.0


def test_divergence_linear_expansion():
    """Scenario 2: Unit Divergence (Linear Expansion). u=x, v=y, w=z -> div=3.0."""
    # Gradient: du/dx=1, dv/dy=1, dw/dz=1
    block = setup_divergent_stencil(lambda i, j, k: i, lambda i, j, k: j, lambda i, j, k: k)
    result = compute_local_divergence_v_star(block)
    # Result: (2-0)/2 + (2-0)/2 + (2-0)/2 = 3.0
    assert math.isclose(result, 3.0, rel_tol=1e-12)


def test_divergence_solenoidal_flow():
    """Scenario 3: Solenoidal Field (Rotational). u=y, v=-x, w=0 -> div=0."""
    block = setup_divergent_stencil(lambda i, j, k: j, lambda i, j, k: -i, lambda i, j, k: 0)
    result = compute_local_divergence_v_star(block)
    assert abs(result) < 1e-12


def test_divergence_asymmetric_flow():
    """Scenario 4: Asymmetric Gradient. u=2x, v=0, w=0 -> div=2.0."""
    block = setup_divergent_stencil(lambda i, j, k: 2 * i, lambda i, j, k: 0, lambda i, j, k: 0)
    result = compute_local_divergence_v_star(block)
    assert result == 2.0


# --- FORENSIC LOGGER TESTS (RULE 7) ---

def test_divergence_topology_crash_logger(caplog):
    """Verify CRITICAL log on missing neighbor (AttributeError)."""
    block = make_step3_output_dummy()
    object.__setattr__(block, '_i_minus', None)  # Break the stencil
    
    with caplog.at_level(logging.CRITICAL), pytest.raises(AttributeError):
        compute_local_divergence_v_star(block)
            
    assert "TOPOLOGY CRASH" in caplog.text


def test_divergence_geometry_crash_logger(caplog):
    """Verify CRITICAL log on zero dimension (ZeroDivisionError)."""
    block = setup_divergent_stencil(lambda i, j, k: i, lambda i, j, k: j, lambda i, j, k: k)
    object.__setattr__(block, '_dx', 0.0)  # Break the physics
    
    with caplog.at_level(logging.CRITICAL), pytest.raises(ZeroDivisionError):
        compute_local_divergence_v_star(block)
            
    assert "GEOMETRY CRASH" in caplog.text


def test_divergence_instability_logger(caplog):
    """Verify ERROR log on non-finite result (ArithmeticError)."""
    # Create an infinite gradient
    block = setup_divergent_stencil(lambda i, j, k: i * np.inf, lambda i, j, k: 0, lambda i, j, k: 0)
    
    with caplog.at_level(logging.ERROR), pytest.raises(ArithmeticError, match="PPE source term is poisoned"):
        compute_local_divergence_v_star(block)
            
    assert "MATH FAILURE" in caplog.text
    assert "Components:" in caplog.text
