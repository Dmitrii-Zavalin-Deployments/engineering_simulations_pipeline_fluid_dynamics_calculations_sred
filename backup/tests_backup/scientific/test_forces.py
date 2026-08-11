# tests/scientific/test_forces.py

import logging

import numpy as np
import pytest

from src.step3.ops.forces import get_local_body_force
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def setup_force_block(block, force_tuple):
    """
    Manually wires the force vector into the StencilBlock's protected slot.
    Mimics Step 2 Assembly logic to satisfy Rule 4 (Hierarchy over Convenience).
    """
    object.__setattr__(block, '_f_vals', force_tuple)
    return block


# --- PHYSICS ACCURACY GATE ---

def test_body_force_zero_gravity():
    """Scenario 1: The Vacuum Gate. v=0 configuration returns null vector."""
    block = make_step3_output_dummy()
    expected = (0.0, 0.0, 0.0)
    setup_force_block(block, expected)
    
    result = get_local_body_force(block)
    
    assert result == expected
    assert isinstance(result, tuple)


def test_body_force_earth_gravity():
    """Scenario 2: Standard Vertical Force. Fy=-9.81 mapping verification."""
    block = make_step3_output_dummy()
    expected = (0.0, -9.81, 0.0)
    setup_force_block(block, expected)
    
    result = get_local_body_force(block)
    
    assert result == expected
    assert result[1] == -9.81


def test_body_force_precision_integrity():
    """Scenario 3: High Precision Preservation. Verifies Rule 7 machine precision."""
    block = make_step3_output_dummy()
    val = 1.234567890123456
    expected = (val, val, val)
    setup_force_block(block, expected)
    
    result = get_local_body_force(block)
    
    assert result[0] == pytest.approx(val, abs=1e-15)


# --- FORENSIC LOGGING & SAFETY GATE ---

def test_body_force_topology_crash_logger(caplog):
    """Verify CRITICAL log when f_vals attribute is missing (Rule 8)."""
    block = make_step3_output_dummy()
    # Force complete removal of the attribute to trigger AttributeError
    delattr(block, '_f_vals')
    
    with caplog.at_level(logging.CRITICAL), pytest.raises(AttributeError, match="force metadata missing"):
        get_local_body_force(block)
            
    assert "TOPOLOGY CRASH" in caplog.text


def test_body_force_contract_violation_logger(caplog):
    """Verify CRITICAL log when force vector has wrong dimensions (Rule 8)."""
    block = make_step3_output_dummy()
    # Provide a 2D vector instead of 3D
    setup_force_block(block, (1.0, 1.0))
    
    with caplog.at_level(logging.CRITICAL), pytest.raises(ValueError, match="Invalid body force vector"):
        get_local_body_force(block)
            
    assert "CONTRACT VIOLATION" in caplog.text
    assert "Expected 3 components" in caplog.text


def test_body_force_numerical_instability_logger(caplog):
    """Verify ERROR log when forces are non-finite (Rule 7)."""
    block = make_step3_output_dummy()
    # Inject NaN into the force vector
    setup_force_block(block, (1.0, np.nan, 0.0))
    
    with caplog.at_level(logging.ERROR), pytest.raises(ArithmeticError, match="force is non-finite"):
        get_local_body_force(block)
            
    assert "MATH FAILURE" in caplog.text
    assert "F_vals: [" in caplog.text


def test_body_force_success_trace(caplog):
    """Verify DEBUG log on successful retrieval."""
    block = make_step3_output_dummy()
    setup_force_block(block, (0.0, 0.0, -9.81))
    
    with caplog.at_level(logging.DEBUG):
        get_local_body_force(block)
        
    assert "OPS [Success]" in caplog.text
    assert "G-Vector" in caplog.text
