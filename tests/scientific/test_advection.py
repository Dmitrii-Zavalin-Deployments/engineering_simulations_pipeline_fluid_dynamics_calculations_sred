# tests/scientific/test_advection.py

import logging
import math

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step3.ops.advection import (
    compute_local_advection,
    compute_local_advection_vector,
)

# Rule 6/9: Using Step 2/3 dummies to ensure stencil_matrix is fully initialized
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy

# Rule 7: Granular Traceability
logger = logging.getLogger(__name__)


def setup_analytical_stencil(velocity_vec, scalar_func):
    """
    Rule 9 Bridge: Uses the real Step 3 Dummy (Production Cells/Buffer).
    Sets up a 1.0 unit coordinate system for clean analytical verification.
    """
    # make_step3 returns the first block directly
    block = make_step3_output_dummy(nx=4, ny=4, nz=4)
    
    # Rule 5: Deterministic Override for math purity
    object.__setattr__(block, '_dx', 1.0)
    object.__setattr__(block, '_dy', 1.0)
    object.__setattr__(block, '_dz', 1.0)

    # Define a logical 3D layout centered around (1,1,1)
    layout = {
        block.center: (1, 1, 1),
        block.i_plus: (2, 1, 1), block.i_minus: (0, 1, 1),
        block.j_plus: (1, 2, 1), block.j_minus: (1, 0, 1),
        block.k_plus: (1, 1, 2), block.k_minus: (1, 1, 0)
    }
    
    for cell, (i, j, k) in layout.items():
        cell.set_field(FI.P, float(scalar_func(i, j, k)))
        
        if cell == block.center:
            cell.set_field(FI.VX, velocity_vec[0])
            cell.set_field(FI.VY, velocity_vec[1])
            cell.set_field(FI.VZ, velocity_vec[2])
        else:
            val = float(scalar_func(i, j, k))
            cell.set_field(FI.VX, val)
            cell.set_field(FI.VY, 0.0) 
            cell.set_field(FI.VZ, 0.0)
            
    return block


# --- PHYSICS VALIDATION SCENARIOS ---

def test_advection_zero_velocity(caplog):
    """Scenario 1: Null Field (The Zero-Gate). v=0 -> Advection=0."""
    block = setup_analytical_stencil((0.0, 0.0, 0.0), lambda i, j, k: i + j + k)
    with caplog.at_level(logging.DEBUG):
        result = compute_local_advection(block, FI.P)
    assert result == 0.0


def test_advection_linear_fidelity(caplog):
    """Scenario 2: Uniform Velocity & Linear Gradient. v=(1,1,1), grad=(1,1,1) -> 3.0."""
    block = setup_analytical_stencil((1.0, 1.0, 1.0), lambda i, j, k: i + j + k)
    with caplog.at_level(logging.DEBUG):
        result = compute_local_advection(block, FI.P)
    assert math.isclose(result, 3.0, rel_tol=1e-12)


def test_advection_vector_component_isolation(caplog):
    """Scenario 3: Vector Test. u=2, grad(VX)=1 -> Result=(2,0,0)."""
    block = setup_analytical_stencil((2.0, 0.0, 0.0), lambda i, j, k: i)
    with caplog.at_level(logging.DEBUG):
        adv_vec = compute_local_advection_vector(block)
    assert adv_vec[0] == 2.0
    assert adv_vec[1] == 0.0
    assert adv_vec[2] == 0.0
    assert "OPS [Success]" in caplog.text


# --- LOGGING & EDGE CASE GATE (RULE 7 & 8) ---

def test_advection_numerical_instability_logger(caplog):
    """Verify Rule 7: Non-finite values trigger ERROR logs."""
    block = setup_analytical_stencil((np.inf, 0.0, 0.0), lambda i, j, k: i)
    with caplog.at_level(logging.ERROR), pytest.raises(ArithmeticError, match="Advection term exploded"):
        compute_local_advection(block, FI.P)
    assert "MATH FAILURE" in caplog.text


def test_advection_topology_crash_logger(caplog):
    """Verify Rule 7: Missing neighbors trigger CRITICAL logs."""
    block = make_step3_output_dummy()
    object.__setattr__(block, '_i_plus', None)
    with caplog.at_level(logging.CRITICAL), pytest.raises(AttributeError):
        compute_local_advection(block, FI.P)
    assert "TOPOLOGY CRASH" in caplog.text


def test_advection_vector_failure_logger(caplog):
    """
    Verify ERROR log when the advection wrapper fails.
    Compliance: Fixed Type-Mismatch using Step 2 Dummy to ensure stencil_matrix exists.
    """
    # 1. Use Step 2 Dummy which GUARANTEES stencil_matrix is populated
    state = make_step2_output_dummy()
    block = state.stencil_matrix[0]
    
    # 2. Force a failure inside the block
    object.__setattr__(block, '_center', None) 
    
    with caplog.at_level(logging.ERROR), pytest.raises(AttributeError):
        compute_local_advection_vector(block)
            
    assert "OPS [Failure]" in caplog.text
