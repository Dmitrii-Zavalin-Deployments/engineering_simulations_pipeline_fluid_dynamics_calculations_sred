# tests/scientific/test_laplacian.py

import logging
import math

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step3.ops.laplacian import (
    compute_local_laplacian,
    compute_local_laplacian_p_next,
    compute_local_laplacian_v_n,
)
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy

# Rule 7: Granular Traceability
logger = logging.getLogger(__name__)

def setup_analytical_laplacian(field_id, f_func):
    """
    Rule 9 Bridge: Uses the production Step 3 Dummy.
    Sets up a 1.0 unit coordinate system for analytical Laplacian verification.
    """
    # Get a real production block wired to the foundation buffer
    block = make_step3_output_dummy(nx=4, ny=4, nz=4)
    
    # Rule 5: Deterministic Spacing for math purity
    object.__setattr__(block, '_dx', 1.0)
    object.__setattr__(block, '_dy', 1.0)
    object.__setattr__(block, '_dz', 1.0)

    # Centralized layout for finite difference neighbors
    layout = {
        block.center: (1, 1, 1),
        block.i_plus: (2, 1, 1), block.i_minus: (0, 1, 1),
        block.j_plus: (1, 2, 1), block.j_minus: (1, 0, 1),
        block.k_plus: (1, 1, 2), block.k_minus: (1, 1, 0)
    }
    
    for cell, (i, j, k) in layout.items():
        cell.set_field(field_id, float(f_func(i, j, k)))
            
    return block

# --- PHYSICS VALIDATION SCENARIOS ---

def test_laplacian_harmonic_field():
    """Scenario 1: Harmonic Field (∇²f = 0). f = x + y + z."""
    block = setup_analytical_laplacian(FI.P, lambda i, j, k: i + j + k)
    result = compute_local_laplacian(block, FI.P)
    # Linear fields have zero second derivatives
    assert result == 0.0

def test_laplacian_quadratic_field():
    """Scenario 2: Quadratic Field (Constant Laplacian). f = x² + y² + z² -> ∇²f = 6.0."""
    block = setup_analytical_laplacian(FI.P, lambda i, j, k: i**2 + j**2 + k**2)
    result = compute_local_laplacian(block, FI.P)
    # dx=1.0: (4 - 2(1) + 0)/1² = 2.0. Total = 2+2+2 = 6.0
    assert math.isclose(result, 6.0, rel_tol=1e-12)

def test_laplacian_velocity_vector(caplog):
    """Scenario 3: Velocity Vector Laplacian. u=x² -> ∇²u=2."""
    block = setup_analytical_laplacian(FI.VX, lambda i, j, k: i**2)
    # Ensure other components are zeroed
    for cell in [block.center, block.i_plus, block.i_minus]:
        cell.set_field(FI.VY, 0.0)
        cell.set_field(FI.VZ, 0.0)
    
    with caplog.at_level(logging.DEBUG):
        lap_v = compute_local_laplacian_v_n(block)
    
    assert math.isclose(lap_v[0], 2.0, rel_tol=1e-12)
    assert lap_v[1] == 0.0
    assert lap_v[2] == 0.0
    assert "OPS [Start]" in caplog.text

def test_laplacian_p_next_gate():
    """Scenario 4: Pressure Next Gate wrapper verification."""
    block = setup_analytical_laplacian(FI.P_NEXT, lambda i, j, k: j**2)
    result = compute_local_laplacian_p_next(block)
    assert math.isclose(result, 2.0, rel_tol=1e-12)

def test_laplacian_stretched_grid():
    """Scenario 5: Stretched Grid Scaling. dx=2.0, f=x² -> ∇²f=0.5."""
    block = setup_analytical_laplacian(FI.P, lambda i, j, k: i**2)
    object.__setattr__(block, '_dx', 2.0)
    
    result = compute_local_laplacian(block, FI.P)
    # (4 - 2(1) + 0) / (2.0²) = 2/4 = 0.5
    assert math.isclose(result, 0.5, rel_tol=1e-12)

# --- FORENSIC LOGGER TESTS (RULE 7) ---

def test_laplacian_topology_crash_logger(caplog):
    """Verify CRITICAL log on missing neighbor."""
    block = make_step3_output_dummy()
    object.__setattr__(block, '_center', None) # Break the stencil
    
    with caplog.at_level(logging.CRITICAL), pytest.raises(AttributeError):
        compute_local_laplacian(block, FI.P)
            
    assert "TOPOLOGY CRASH" in caplog.text

def test_laplacian_geometry_crash_logger(caplog):
    """Verify CRITICAL log on zero/negative dimension."""
    block = make_step3_output_dummy()
    object.__setattr__(block, '_dz', 0.0)
    
    with caplog.at_level(logging.CRITICAL), pytest.raises(ZeroDivisionError):
        compute_local_laplacian(block, FI.P)
            
    assert "GEOMETRY CRASH" in caplog.text

def test_laplacian_instability_logger(caplog):
    """Verify ERROR log on non-finite result."""
    # Create an explosion via infinite values in the field
    block = setup_analytical_laplacian(FI.P, lambda i, j, k: i * np.inf)
    
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ArithmeticError, match="Laplacian exploded"):
            compute_local_laplacian(block, FI.P)
            
    assert "MATH FAILURE" in caplog.text
    assert "Field: P" in caplog.text

def test_laplacian_vector_failure_logger(caplog):
    """Verify ERROR log when the vector wrapper fails due to topology."""
    block = make_step3_output_dummy()
    object.__setattr__(block, '_center', None) # Force AttributeError
    
    with caplog.at_level(logging.ERROR), pytest.raises(AttributeError):
        compute_local_laplacian_v_n(block)
            
    assert "OPS [Failure]" in caplog.text