# tests/helpers/solver_output_schema_dummy.py

from src.common.solver_state import SolverState
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


def make_output_schema_dummy(nx: int = 4, ny: int = 4, nz: int = 4) -> SolverState:
    """
    Hydrates a terminal SolverState object for Archivist Testing.
    Ensures snapshot paths align with Rule 4 SSoT.
    """
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Reset loop flag to indicate simulation completion
    state.ready_for_time_loop = False
    
    # Update manifest with realistic snapshot names
    # Note: We ensure these match the 'output/' prefix expected by the archiver
    state.manifest.saved_snapshots = [
        "output/snapshot_0000.h5",
        "output/snapshot_0001.h5",
        "output/snapshot_0002.h5"
    ]
    
    return state