# tests/step2/test_orchestrate_step2.py

from unittest.mock import patch

from src.common.cell import Cell
from src.common.stencil_block import StencilBlock
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def test_orchestrate_step2_full_flow_with_debug():
    """
    Verifies orchestration while maintaining Memory Parity (Rule 9).
    We manually build a small matrix using the LIVE state's buffer
    to bypass the Memory Swap and __slots__ issues.
    """
    # 1. Setup the 'Live' state
    state = make_step1_output_dummy(nx=1, ny=1, nz=1)
    state.ready_for_time_loop = False
    
    # 2. Build a valid matrix specifically for THIS state's buffer
    # This guarantees parity for the verify_foundation_integrity POST.
    nx_buf, ny_buf = state.grid.nx + 2, state.grid.ny + 2
    
    # Simple 1x1x1 core cell at flattened index
    # idx = i + nx_buf * (j + ny_buf * k) -> 1 + 3*(1 + 3*1) = 13
    center_cell = Cell(index=13, fields_buffer=state.fields.data, nx_buf=nx_buf, ny_buf=ny_buf, is_ghost=False)
    
    # We only need the center cell populated for the POST check to pass
    block = StencilBlock(
        center=center_cell,
        i_minus=center_cell, i_plus=center_cell,
        j_minus=center_cell, j_plus=center_cell,
        k_minus=center_cell, k_plus=center_cell,
        dx=0.1, dy=0.1, dz=0.1, dt=0.01, rho=1.0, mu=0.1, f_vals=(0,0,0)
    )
    valid_matrix = [block]

    # 3. Patch the assembler to return our buffer-aligned matrix
    with patch("src.step2.orchestrate_step2.assemble_stencil_matrix") as mock_assembler:
        mock_assembler.return_value = valid_matrix
        
        with patch("src.step2.orchestrate_step2.DEBUG", True):
            result_state = orchestrate_step2(state)
            
            assert result_state.ready_for_time_loop is True
            assert result_state.stencil_matrix == valid_matrix
            mock_assembler.assert_called_once_with(state)

def test_orchestrate_step2_standard_flow():
    """
    Standard flow test without DEBUG, using the same logic.
    """
    state = make_step1_output_dummy(nx=1, ny=1, nz=1)
    nx_buf, ny_buf = 3, 3
    center_cell = Cell(13, state.fields.data, nx_buf, ny_buf, False)
    block = StencilBlock(center_cell, center_cell, center_cell, center_cell, center_cell, center_cell, center_cell, 
                         0.1, 0.1, 0.1, 0.01, 1.0, 0.1, (0,0,0))
    
    with patch("src.step2.orchestrate_step2.assemble_stencil_matrix") as mock_assembler:
        mock_assembler.return_value = [block]
        orchestrate_step2(state)
        assert state.ready_for_time_loop is True