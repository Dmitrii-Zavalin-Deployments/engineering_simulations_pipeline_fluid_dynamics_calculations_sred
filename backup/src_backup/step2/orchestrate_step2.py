# src/step2/orchestrate_step2.py

from src.common.solver_state import SolverState
from src.step2.stencil_assembler import assemble_stencil_matrix

# Rule 7: Granular Traceability
DEBUG = False

def orchestrate_step2(state: SolverState) -> SolverState:
    """
    Orchestrates the construction of the Stencil Matrix.
    """
    if DEBUG:
        print("DEBUG [Step 2.0]: Orchestration Started")

    # The registry is now encapsulated within the assembler, 
    # ensuring a clean lifecycle for every simulation run.
    state.stencil_matrix = assemble_stencil_matrix(state)
    
    state.ready_for_time_loop = True
    
    if DEBUG:
        print("DEBUG [Step 2.0]: Orchestration Finalized.")
    
    return state