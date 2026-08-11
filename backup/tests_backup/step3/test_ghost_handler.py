# tests/step3/test_ghost_handler.py

from unittest.mock import MagicMock

import pytest

from src.step3.ops.ghost_handler import sync_ghost_trial_buffers


def test_sync_ghost_trial_buffers_success():
    """
    Targets Lines 28-36: Verifies standard synchronization of trial buffers.
    """
    # 1. Mock a Cell with the required Rule 9 properties
    mock_cell = MagicMock()
    mock_cell.vx = 1.0
    mock_cell.vy = 2.0
    mock_cell.vz = 3.0
    mock_cell.p = 101325.0
    
    # 2. Mock the StencilBlock
    mock_block = MagicMock()
    mock_block.id = "test_block_001"
    mock_block.center = mock_cell
    
    # 3. Execute sync
    sync_ghost_trial_buffers(mock_block)
    
    # 4. Verify the trial buffers (STAR/NEXT) were updated correctly
    assert mock_cell.vx_star == 1.0
    assert mock_cell.vy_star == 2.0
    assert mock_cell.vz_star == 3.0
    assert mock_cell.p_next == 101325.0

def test_sync_ghost_trial_buffers_contract_violation():
    """
    Targets Lines 38-44: Triggers AttributeError to verify structural failure handling.
    """
    # 1. Create a "malformed" cell that lacks a required property (e.g., vx_star)
    class MalformedCell:
        def __init__(self):
            self.vx = 1.0
            # Missing vx_star, vy_star, etc.
            
    mock_block = MagicMock()
    mock_block.id = "malformed_block_999"
    mock_block.center = MalformedCell()
    
    # 2. Verify that a RuntimeError is raised per Rule 5
    with pytest.raises(RuntimeError, match="Ghost sync failed: Cell foundation is malformed."):
        sync_ghost_trial_buffers(mock_block)