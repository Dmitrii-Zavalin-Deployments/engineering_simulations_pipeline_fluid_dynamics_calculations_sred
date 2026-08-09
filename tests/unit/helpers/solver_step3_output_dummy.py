# tests/helpers/solver_step3_output_dummy.py

from src.common.field_schema import FI
from src.step2.stencil_assembler import assemble_stencil_matrix
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def make_step3_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4, block_index: int = 0):
    """
    Step 3 Output: The 'Trial' state.
    Bridges the gap between monolithic SolverState and StencilBlock arrays.
    """
    # 1. Generate the foundation state from Step 2
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    data = state.fields.data
    
    # 2. Assign Trial Data
    data[:, FI.VX_STAR] = 0.50 
    data[:, FI.VY_STAR] = 0.50
    data[:, FI.VZ_STAR] = 0.50
    data[:, FI.P_NEXT]  = 0.012
    
    # 3. FIX: Use the correct function name and the single 'state' argument
    # This wires the Cell objects in the StencilBlocks to the updated data buffer
    state.stencil_matrix = assemble_stencil_matrix(state)
    
    return state.stencil_matrix[block_index]