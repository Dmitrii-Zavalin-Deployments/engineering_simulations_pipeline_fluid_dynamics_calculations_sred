# tests/helpers/solver_step2_output_dummy.py

"""
Archivist Testing: Snapshot-based Test Baseline (Step 2).

Compliance:
- Rule 6: Zero-Redundancy (Uses production Cell for 100% parity).
- Rule 7: Atomic Numerical Truth (Fixed data for verification).
- Rule 9: Sentinel Integrity (Real pointers into monolithic foundation).
"""

from src.common.cell import Cell  # Mandate: Use production code over Mocks
from src.common.stencil_block import StencilBlock
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def make_step2_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Fixed Step 2 Dummy: Complies with Rule 7 (Offset 1 Topology).
    Maps Core [0...nx-1] to Memory Indices [1...nx].
    Uses the production Cell class to ensure no technical debt drift.
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    physics_params = {
        "dx": 0.25, "dy": 0.25, "dz": 0.25,
        "dt": 0.001, "rho": 1000.0, "mu": 0.001,
        "f_vals": (0.0, 0.0, -9.81)
    }
    
    state.stencil_matrix = []
    # Rule 7.2: 4x4x4 core maps to 6x6x6 memory structure
    nx_buf, ny_buf, _ = nx + 2, ny + 2, nz + 2
    buffer = state.fields.data
    
    def get_idx(i, j, k):
        # Flattening logic: i + (j * width) + (k * width * height)
        return i + nx_buf * (j + ny_buf * k)

    # Core loop: Memory Indices 1 to nx
    for k in range(1, nz + 1):
        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                
                # Instantiate REAL production Cells
                # Parameters: index, fields_buffer, nx_buf, ny_buf, is_ghost
                cell_c = Cell(get_idx(i, j, k), buffer, nx_buf, ny_buf, False)
                
                # Neighbors pull from the Unified Foundation (Ghosts at 0 and n+1)
                cell_im = Cell(get_idx(i-1, j, k), buffer, nx_buf, ny_buf, (i-1 == 0))
                cell_ip = Cell(get_idx(i+1, j, k), buffer, nx_buf, ny_buf, (i+1 == nx+1))
                
                cell_jm = Cell(get_idx(i, j-1, k), buffer, nx_buf, ny_buf, (j-1 == 0))
                cell_jp = Cell(get_idx(i, j+1, k), buffer, nx_buf, ny_buf, (j+1 == ny+1))
                
                cell_km = Cell(get_idx(i, j, k-1), buffer, nx_buf, ny_buf, (k-1 == 0))
                cell_kp = Cell(get_idx(i, j, k+1), buffer, nx_buf, ny_buf, (k+1 == nz+1))
                
                block = StencilBlock(
                    center=cell_c,
                    i_minus=cell_im, i_plus=cell_ip,
                    j_minus=cell_jm, j_plus=cell_jp,
                    k_minus=cell_km, k_plus=cell_kp,
                    **physics_params
                )
                state.stencil_matrix.append(block)
    
    state.ready_for_time_loop = True
    return state