# tests/common/test_elasticity_manager.py

from types import SimpleNamespace

import numpy as np
import pytest

from src.common.elasticity import ElasticManager
from src.common.field_schema import FI

# ----------------------------------------------------------------
# 1. MOCK OBJECTS & FIXTURES
# ----------------------------------------------------------------

class MockConfig:
    def __init__(self):
        # Aligned with ElasticManager requirements
        self.dt_min_limit = 0.001
        self.ppe_max_retries = 3

@pytest.fixture
def config():
    return MockConfig()

@pytest.fixture
def state_mock():
    """
    Creates a mock state that satisfies the SSoT requirements of ElasticManager.
    Includes simulation_parameters and a placeholder for audit_physical_bounds.
    """
    data = np.zeros((10, 20)) 
    fields = SimpleNamespace(data=data)
    sim_params = SimpleNamespace(time_step=0.5)
    
    state = SimpleNamespace(
        fields=fields, 
        simulation_parameters=sim_params,
        iteration=0,
        time=0.0
    )
    
    # Mock the audit method to prevent errors during commit tests
    state.audit_physical_bounds = lambda: None 
    return state

# ----------------------------------------------------------------
# 2. THE INTEGRATED TEST SUITE
# ----------------------------------------------------------------

def test_safety_ladder_initialization(config, state_mock):
    """Verifies Rule 5: Ladder is correctly pre-calculated from SSoT."""
    manager = ElasticManager(config, state_mock)
    # Range should be [0.5, 0.333..., 0.166..., 0.001]
    assert len(manager._dt_range) == config.ppe_max_retries + 1
    assert manager._dt_range[0] == 0.5
    assert manager._dt_range[-1] == pytest.approx(config.dt_min_limit)

def test_stabilization_descent_logic(config, state_mock):
    """Verifies that is_needed=True correctly moves down the ladder."""
    manager = ElasticManager(config, state_mock)
    initial_dt = manager.dt
    
    manager.stabilization(is_needed=True)
    
    assert manager.dt < initial_dt
    assert manager._iteration == 1
    assert manager.dt == manager._dt_range[1]

def test_stabilization_exhaustion_error(config, state_mock):
    """Verifies RuntimeError triggers when the ladder floor is hit."""
    manager = ElasticManager(config, state_mock)

    # Exhaust the 3 retries
    with pytest.raises(RuntimeError, match="CRITICAL INSTABILITY"):
        for _ in range(config.ppe_max_retries + 1):
            manager.stabilization(is_needed=True)

def test_stabilization_recovery_logic(config, state_mock):
    """Verifies is_needed=False resets the ladder and commits data."""
    manager = ElasticManager(config, state_mock)
    
    # 1. Descend to the bottom of the ladder
    manager.stabilization(is_needed=True)
    assert manager.dt < 0.5
    
    # 2. Trigger success (is_needed=False)
    # This calls validate_and_commit() internally
    manager.stabilization(is_needed=False)
    
    # 3. Verify Reset
    assert manager.dt == 0.5
    assert manager._iteration == 0

def test_validate_and_commit_data_transfer(config, state_mock):
    """Verifies Rule 9: Trial (_STAR) data is promoted to Foundation fields."""
    manager = ElasticManager(config, state_mock)
    
    # Set trial values
    state_mock.fields.data[:, FI.VX_STAR] = 1.23
    state_mock.fields.data[:, FI.P_NEXT] = 4.56
    
    # Manual commit check
    manager.validate_and_commit()
    
    # Verify foundation updated
    assert np.all(state_mock.fields.data[:, FI.VX] == 1.23)
    assert np.all(state_mock.fields.data[:, FI.P] == 4.56)
    assert state_mock.iteration == 1
    assert state_mock.time == 0.5

def test_audit_failure_propagation(config, state_mock):
    """Verifies that an audit failure blocks the data commit."""
    manager = ElasticManager(config, state_mock)
    
    # Override mock to simulate a physics violation (e.g., NaN detected)
    def fail_audit():
        raise ArithmeticError("Physics Violation")
    state_mock.audit_physical_bounds = fail_audit
    
    # Try to stabilize (success signal)
    with pytest.raises(ArithmeticError, match="Physics Violation"):
        manager.stabilization(is_needed=False)
        
    # Verify iteration did not increment because commit was blocked
    assert state_mock.iteration == 0