# tests/step1/test_orchestrate_step1.py

from unittest.mock import MagicMock, patch

import numpy as np

from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import create_validated_input


def test_orchestrate_step1_debug_path():
    """
    Targets lines 35, 102-103 in src/step1/orchestrate_step1.py.
    Uses validated dummy input to verify the full assembly path.
    """
    # 1. Setup using your Protocol-compliant Dummy
    # This automatically provides valid types like "no-slip" and "x_min"
    mock_context = MagicMock()
    mock_context.input_data = create_validated_input(nx=2, ny=2, nz=2)
    
    # 2. Ensure the reference_velocity branch is skipped for this path
    # (The dummy has reference_velocity, but we can delete the private flag to target coverage)
    if hasattr(mock_context.input_data.domain_configuration, '_reference_velocity'):
        del mock_context.input_data.domain_configuration._reference_velocity

    # 3. Patch DEBUG and run orchestration
    with patch("src.step1.orchestrate_step1.DEBUG", True):
        state = orchestrate_step1(mock_context)
        
        # 4. Integrity Checks
        assert state is not None
        # 2x2x2 internal + ghost cells = 4x4x4 = 64
        assert state.fields.data.shape[0] == 64
        assert state.domain_configuration.type == "INTERNAL"
        # Verify boundary conditions were mapped (dummy provides 7)
        assert len(state.boundary_conditions.conditions) == 7

def test_orchestrate_step1_reference_velocity_branch():
    """
    Targets lines 51-52: Ensure reference_velocity is assigned if explicitly present.
    """
    # 1. Setup with dummy (which contains a reference_velocity by default)
    mock_context = MagicMock()
    mock_context.input_data = create_validated_input(nx=1, ny=1, nz=1)
    
    # Ensure the branch is triggered
    ref_vel = [5.0, 0.0, 0.0]
    mock_context.input_data.domain_configuration._reference_velocity = ref_vel
    mock_context.input_data.domain_configuration.reference_velocity = ref_vel

    # 2. Run orchestration
    state = orchestrate_step1(mock_context)
    
    # 3. Verify logic
    assert isinstance(state.domain_configuration.reference_velocity, np.ndarray)
    assert np.array_equal(state.domain_configuration.reference_velocity, np.array(ref_vel))