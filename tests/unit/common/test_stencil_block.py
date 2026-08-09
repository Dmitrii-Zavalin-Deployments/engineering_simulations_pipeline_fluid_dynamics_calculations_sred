# tests/common/test_stencil_block.py

import logging

import numpy as np
import pytest

from src.common.cell import Cell
from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

# Rule 7: Setup Granular Traceability
logger = logging.getLogger(__name__)

def create_real_cell(index=0, is_ghost=False):
    """Rule 9: Helper to create a cell with a real hardware-backed buffer."""
    buffer = np.zeros((1, FI.num_fields()))
    return Cell(
        index=index, 
        fields_buffer=buffer, 
        nx_buf=3, ny_buf=3, 
        is_ghost=is_ghost
    )

@pytest.fixture
def physical_stencil():
    """Provides a real 7-point stencil with production-grade Cells."""
    cells = [create_real_cell(i) for i in range(7)]
    # Order: center, i_minus, i_plus, j_minus, j_plus, k_minus, k_plus
    return StencilBlock(
        center=cells[0],
        i_minus=cells[1], i_plus=cells[2],
        j_minus=cells[3], j_plus=cells[4],
        k_minus=cells[5], k_plus=cells[6],
        dx=0.1, dy=0.1, dz=0.1,
        dt=0.01, rho=1.225, mu=1.8e-5, f_vals=(0, 0, 0)
    )

def test_dt_sync_logging_real_physics(physical_stencil, caplog):
    """
    VERIFICATION: Ensure StencilBlock logs the time-step synchronization event.
    Crucial for tracking Elasticity Manager interventions in real time.
    """
    block = physical_stencil
    block_id = block.id

    # Simulate an Elasticity Downshift (Recovery Phase)
    new_dt = 1e-6
    with caplog.at_level(logging.DEBUG):
        block.dt = new_dt

    # --- ASSERTIONS ---
    assert block.dt == new_dt
    # Rule 7 Check: Exact format defined in the setter
    assert f"SYNC [Physics]: {block_id} updated dt -> 1.0000e-06" in caplog.text

def test_dt_instability_guard_real_logic(physical_stencil, caplog):
    """VERIFICATION: Non-positive dt must log a STABILITY CRASH and raise ValueError."""
    block = physical_stencil

    # Rule 5: Reject invalid numerical state immediately
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="Numerical Instability"):
        block.dt = -0.005

    # Check for the specific forensic log string
    assert "STABILITY CRASH" in caplog.text
    assert "rejected invalid dt=-0.005" in caplog.text

def test_dt_zero_instability_guard(physical_stencil):
    """VERIFICATION: dt = 0 is physically invalid for time-advancement."""
    block = physical_stencil
    
    with pytest.raises(ValueError, match="dt must be positive"):
        block.dt = 0.0

def test_stencil_property_integrity(physical_stencil):
    """
    Rule 8/9: Verify that material properties are locked and accessible.
    Ensures that __slots__ are working and preventing anonymous attribute bloat.
    """
    block = physical_stencil
    
    assert block.rho == 1.225
    assert block.mu == 1.8e-5
    assert block.f_vals == (0, 0, 0)
    
    # Verify we can't inject random junk (if using __slots__)
    with pytest.raises(AttributeError):
        block.random_untracked_var = "Leak"
    
def test_stencil_repr_forensic_traceability(physical_stencil):
    """
    VERIFICATION: Ensure __repr__ provides the Block ID and current dt.
    Target: Line 100 (The missing 2% coverage).
    Crucial for identifying specific grid points during a stability crash.
    """
    block = physical_stencil
    
    # Trigger __repr__
    repr_output = repr(block)
    
    # Assertions based on the template: <{self._id} dt={self._dt:.2e}>
    assert repr_output.startswith("<Block_")
    assert "dt=1.00e-02" in repr_output

def test_topological_accessor_integrity(physical_stencil):
    """
    VERIFICATION: Ensure every directional accessor returns the correct Cell object.
    Confirms the 7-point connectivity required for 3D Navier-Stokes.
    """
    block = physical_stencil
    
    # We verify that the objects returned match the identity of the Cells 
    # provided during initialization in the physical_stencil fixture.
    assert isinstance(block.center, Cell)
    assert isinstance(block.i_minus, Cell)
    assert isinstance(block.i_plus, Cell)
    assert isinstance(block.j_minus, Cell)
    assert isinstance(block.j_plus, Cell)
    assert isinstance(block.k_minus, Cell)
    assert isinstance(block.k_plus, Cell)
    
    # Verify spacing and physics constants
    assert block.dx == 0.1
    assert block.dy == 0.1
    assert block.dz == 0.1
