# tests/step3/test_ghost_integrity.py

import logging
import math

import numpy as np
import pytest

from src.common.cell import Cell
from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.ops.ghost_handler import sync_ghost_trial_buffers

# Rule 7: Granular Traceability - Setup logger for the Clean Room
logger = logging.getLogger(__name__)

def create_identity_primed_cell(index: int, nx: int = 3, ny: int = 3) -> Cell:
    """
    Rule 9: Identity Priming Strategy.
    Fills the buffer with: Value = Index + (Field_ID / 10.0)
    Example: Index 5, Field VX (0) -> 5.0 | Field P (6) -> 5.6
    """
    buffer = np.zeros((1, FI.num_fields()))
    for field in FI:
        buffer[0, field] = float(index) + (float(field) / 10.0)
    
    return Cell(
        index=0, # Local index in this 1-row buffer
        fields_buffer=buffer, 
        nx_buf=nx, 
        ny_buf=ny, 
        is_ghost=True
    )

@pytest.fixture
def identity_block():
    """Constructs a real StencilBlock using the Identity Priming Strategy."""
    center = create_identity_primed_cell(index=50) # Representative index
    
    # Poison the Trial Buffers (Rule 7: Physics is guilty)
    center.fields_buffer[0, FI.VX_STAR] = np.nan
    center.fields_buffer[0, FI.P_NEXT]  = np.inf
    
    # Neighbor dummies to satisfy Rule 5 (Deterministic Init)
    nb = [create_identity_primed_cell(i) for i in range(6)]
    
    return StencilBlock(
        center=center,
        i_minus=nb[0], i_plus=nb[1],
        j_minus=nb[2], j_plus=nb[3],
        k_minus=nb[4], k_plus=nb[5],
        dx=0.1, dy=0.1, dz=0.1, dt=0.01,
        rho=1.225, mu=1.8e-5, f_vals=(0, 0, 0)
    )

def test_ghost_sync_atomic_precision(identity_block, caplog):
    """
    STS Rule 7: Verify that sync_ghost_trial_buffers recovers 
    Identity-Primed values to machine precision.
    """
    block = identity_block
    center = block.center
    
    # Pre-Flight Check: Confirm poison is active
    assert np.isnan(center.vx_star)
    
    # Execute the Operator
    with caplog.at_level(logging.DEBUG):
        sync_ghost_trial_buffers(block)
    
    # Post-Flight Verification: Value = Index + (Field_ID / 10.0)
    # Expected VX_STAR (ID 3) should now match Foundation VX (ID 0)
    # Foundation VX for index 50 is 50.0
    expected_val = 50.0 
    
    assert math.isclose(center.vx_star, expected_val, rel_tol=1e-12)
    assert math.isclose(center.vy_star, 50.1, rel_tol=1e-12)
    assert math.isclose(center.vz_star, 50.2, rel_tol=1e-12)
    assert math.isclose(center.p_next,  50.6, rel_tol=1e-12) # Foundation P is 50.6

    # Rule 7: Log Audit
    assert "GHOST SYNC [Success]" in caplog.text
    logger.info(f"STS PASSED: Block {block.id} recovered atomic identity {expected_val}")

def test_ghost_sync_memory_isolation(identity_block):
    """
    Rule 9: Ensure no 'Anonymous Memory Swaps' occurred.
    Foundation must remain EXACTLY Index + (Field_ID / 10.0).
    """
    block = identity_block
    sync_ghost_trial_buffers(block)
    
    # If the operator accidentally swapped pointers instead of values, 
    # the Foundation P (50.6) would be corrupted.
    assert block.center.p == 50.6
    assert block.center.vx == 50.0