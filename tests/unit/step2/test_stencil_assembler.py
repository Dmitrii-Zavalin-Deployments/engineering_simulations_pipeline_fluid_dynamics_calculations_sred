# tests/step2/test_stencil_assembler.py

import logging

import numpy as np
import pytest

from src.step2.stencil_assembler import CellRegistry, assemble_stencil_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def get_matrix_3d(stencil_list):
    """
    Helper to map the flat stencil list back to a 3D coordinate lookup.
    Note: We access the center cell's internal index shift logic to find coords.
    """
    # Using a dictionary with (i, j, k) tuples as keys for easy testing.
    # Data is mapped based on the logical center coordinate attributes.
    return {(b.center.i, b.center.j, b.center.k): b for b in stencil_list}

def test_stencil_assembly_logic():
    """Verify topology audit: Core assembly count and ghost zone reach."""
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    stencil_list = assemble_stencil_matrix(state)
    
    # 1. Topology Audit: Core Assembly Count
    # Stencils are only created for the CORE domain [0, nx-1]
    expected_count = nx * ny * nz
    assert len(stencil_list) == expected_count, f"Assembly count mismatch: {len(stencil_list)} != {expected_count}"
    
    # Verify a sample block (0,0,0)
    # Based on the k,j,i loop order, (0,0,0) is the first element.
    sample_block = stencil_list[0] 
    
    # 2. Verify Center is Core
    assert sample_block.center.is_ghost is False
    
    # 3. Verify Ghost Neighbor: (0,0,0) neighbor at (-1, 0, 0)
    # This proves the assembler correctly resolves pointers into the Ghost Zone.
    assert sample_block.i_minus.is_ghost is True
    assert sample_block.i_minus.index < sample_block.center.index

def test_stencil_physics_consistency():
    """
    Verify Rule 4 (SSoT) & Rule 5: 
    Ensure global solver state is correctly injected into every block via sub-containers.
    """
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Set unique values to ensure they aren't defaults
    target_dt = 0.0123
    target_rho = 999.0
    target_f = (0.1, 0.2, 0.3)
    
    # Direct container access to avoid facade properties (Rule 4 alignment)
    state.simulation_parameters._time_step = target_dt
    state.fluid_properties._density = target_rho
    state.external_forces._force_vector = np.array(target_f)
    
    stencil_list = assemble_stencil_matrix(state)
    
    for block in stencil_list:
        assert block.dt == target_dt
        assert block.rho == target_rho
        # Convert tuple for comparison against StencilBlock's internal representation
        assert tuple(block.f_vals) == target_f

def test_stencil_pointer_integrity():
    """
    Verify that adjacent blocks share the same Cell instance for their interface.
    This validates the CellRegistry flyweight pattern (Rule 0 / Rule 9).
    """
    nx, ny, nz = 3, 3, 3
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    stencil_list = assemble_stencil_matrix(state)
    
    # Block A at (0,0,0), Block B at (1,0,0)
    # In a k,j,i loop, (0,0,0) is index 0, (1,0,0) is index 1
    block_a = stencil_list[0]
    block_b = stencil_list[1]
    
    # The 'Right' (i_plus) of A must be the 'Center' of B (memory identity)
    assert block_a.i_plus is block_b.center
    # The 'Left' (i_minus) of B must be the 'Center' of A (memory identity)
    assert block_b.i_minus is block_a.center

def test_registry_cache_hit():
    """Direct validation of the CellRegistry flyweight pattern and identity."""
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    registry = CellRegistry(nx, ny, nz)
    
    # Request same coordinate twice
    cell_1 = registry.get_or_create(0, 0, 0, state)
    cell_2 = registry.get_or_create(0, 0, 0, state)
    
    # 'is' checks for memory identity (the same pointer in memory)
    assert cell_1 is cell_2, "Registry failed to return cached instance."

    # Request different coordinate
    cell_3 = registry.get_or_create(1, 0, 0, state)
    assert cell_1 is not cell_3

def test_registry_bounds_enforcement():
    """Verify Rule 7: Registry blocks illegal padding access."""
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    registry = CellRegistry(nx, ny, nz)
    
    # Valid range per Rule 7: [-1, nx]
    # Invalid: -2, nx + 1
    
    with pytest.raises(IndexError, match="out-of-bounds"):
        registry.get_or_create(-2, 0, 0, state)
        
    with pytest.raises(IndexError, match="out-of-bounds"):
        registry.get_or_create(nx + 1, 0, 0, state)

def test_stencil_assembly_logging(caplog):
    """
    Verify Rule 7: Granular Traceability.
    Ensures assembly lifecycle is captured in logs for audit compliance.
    """
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Set log level to INFO to capture the assembly lifecycle
    with caplog.at_level(logging.INFO):
        assemble_stencil_matrix(state)
        
    # Verify presence of the Rule 7 audit trail
    assert "Stencil Assembly Started" in caplog.text
    assert "Successfully assembled 8 Core StencilBlocks" in caplog.text

def test_registry_allocation_logging(caplog):
    """Verify Rule 7: Allocation events trigger diagnostic logs."""
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    registry = CellRegistry(nx, ny, nz)
    
    with caplog.at_level(logging.DEBUG):
        # First call: Triggers an allocation log
        registry.get_or_create(0, 0, 0, state)
        assert "Allocated new cell at (0, 0, 0)" in caplog.text
        
        caplog.clear() # Reset log buffer
        
        # Second call: Cache hit must NOT trigger an allocation log
        registry.get_or_create(0, 0, 0, state)
        assert "Allocated new cell at (0, 0, 0)" not in caplog.text