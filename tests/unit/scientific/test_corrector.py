# tests/scientific/test_corrector.py

import logging

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step3.corrector import apply_local_velocity_correction
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def setup_integration_block(block, dt=1.0, rho=1.0):
    """
    Standardizes the block for analytical testing by forcing unit geometry.
    Uses object.__setattr__ to bypass Read-Only properties.
    """
    params = {
        '_dt': float(dt),
        '_rho': float(rho),
        '_dx': 1.0,
        '_dy': 1.0,
        '_dz': 1.0
    }
    for attr, val in params.items():
        object.__setattr__(block, attr, val)
        
    # Reset the shared buffer for this block's neighborhood to prevent drift
    cells = [block.center, block.i_plus, block.i_minus, 
             block.j_plus, block.j_minus, block.k_plus, block.k_minus]
    for cell in cells:
        cell.fields_buffer[cell.index, :] = 0.0
        
    return block


# --- PHYSICS PROJECTION TESTS ---

def test_corrector_zero_gradient_preservation():
    """Scenario 1: Uniform Pressure. v_next should equal v_star exactly."""
    block = setup_integration_block(make_step3_output_dummy())
    
    block.center.set_field(FI.VX_STAR, 1.0)
    block.center.set_field(FI.VY_STAR, 0.5)
    block.center.set_field(FI.VZ_STAR, 0.2)
    
    # Uniform pressure field -> Grad P = 0
    for cell in [block.center, block.i_plus, block.i_minus, 
                 block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
        cell.set_field(FI.P_NEXT, 10.0)
        
    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX_STAR) == 1.0
    assert block.center.get_field(FI.VY_STAR) == 0.5
    assert block.center.get_field(FI.VZ_STAR) == 0.2


def test_corrector_analytical_correction():
    """
    Scenario 2: Analytical Projection.
    Formula: v = v* - (dt/rho) * grad(P)
    Setup: v*=1.0, dt=0.1, rho=1.0, grad_p=1.0
    Expected: 1.0 - (0.1/1.0)*1.0 = 0.9
    """
    block = setup_integration_block(make_step3_output_dummy(), dt=0.1, rho=1.0)
    
    # Setup Gradient P_x = 1.0 via central difference: (2-0)/2 = 1.0
    block.i_plus.set_field(FI.P_NEXT, 2.0)
    block.i_minus.set_field(FI.P_NEXT, 0.0)
    
    block.center.set_field(FI.VX_STAR, 1.0)
    
    apply_local_velocity_correction(block)
    
    obtained = block.center.get_field(FI.VX_STAR)
    assert obtained == pytest.approx(0.9, abs=1e-15)


def test_corrector_3d_vector_alignment():
    """Scenario 3: Checks simultaneous correction of all 3 components."""
    block = setup_integration_block(make_step3_output_dummy(), dt=1.0, rho=1.0)
    
    block.center.set_field(FI.VX_STAR, 0.0)
    block.center.set_field(FI.VY_STAR, 0.0)
    block.center.set_field(FI.VZ_STAR, 0.0)
    
    # Setup Grad P = (1, 1, 1)
    block.i_plus.set_field(FI.P_NEXT, 2.0); block.i_minus.set_field(FI.P_NEXT, 0.0)
    block.j_plus.set_field(FI.P_NEXT, 2.0); block.j_minus.set_field(FI.P_NEXT, 0.0)
    block.k_plus.set_field(FI.P_NEXT, 2.0); block.k_minus.set_field(FI.P_NEXT, 0.0)
    
    apply_local_velocity_correction(block)
    
    assert block.center.get_field(FI.VX_STAR) == -1.0
    assert block.center.get_field(FI.VY_STAR) == -1.0
    assert block.center.get_field(FI.VZ_STAR) == -1.0


# --- FORENSIC LOGGING & AUDIT TESTS ---

def test_corrector_success_audit_log(caplog):
    """Verifies DEBUG log on successful completion."""
    block = setup_integration_block(make_step3_output_dummy())
    
    with caplog.at_level(logging.DEBUG):
        apply_local_velocity_correction(block)
        
    assert "CORRECT [Success]" in caplog.text
    assert f"Block {block.id}" in caplog.text


def test_corrector_array_leak_detection_log(caplog):
    """
    Verifies Rule 7: DNA Audit.
    If math promotes a result to an array, the corrector must log and recover.
    """
    block = setup_integration_block(make_step3_output_dummy())
    
    # Force a 'leaky' result by mocking the VX_STAR field to return a NumPy array
    # We use a 1D array with one element to simulate the most common leak
    block.center.set_field(FI.VX_STAR, np.array([1.0])) 
    
    with caplog.at_level(logging.DEBUG):
        apply_local_velocity_correction(block)
        
    assert "CORRECT [Success]" in caplog.text


def test_corrector_instability_crash_log(caplog):
    """Verifies ERROR log and ArithmeticError on non-finite velocity (Rule 7)."""
    block = setup_integration_block(make_step3_output_dummy())
    
    # Inject NaN into the intermediate velocity
    block.center.set_field(FI.VX_STAR, np.nan)
    
    with caplog.at_level(logging.ERROR), pytest.raises(ArithmeticError, match="Instability detected"):
        apply_local_velocity_correction(block)
            
    assert "CORRECTOR CRITICAL" in caplog.text
    assert "Non-finite velocity" in caplog.text
