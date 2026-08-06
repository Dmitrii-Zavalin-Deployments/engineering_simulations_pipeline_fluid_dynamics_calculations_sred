# src/step2/factory.py

from src.common.cell import Cell
from src.common.grid_math import get_flat_index
from src.common.solver_state import SolverState

# Rule 7: Granular Traceability
DEBUG = False 

# Explicit constants for ghost cell initialization
GHOST_VELOCITY = (0.0, 0.0, 0.0)
GHOST_PRESSURE = 0.0
GHOST_MASK = 0

def get_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """
    Allocates a Cell instance with strict 2-Tier Topology validation.
    """
    grid = state.grid
    
    # 7.1 Compliance: Define the two valid operational zones
    is_core = (0 <= i < grid.nx) and (0 <= j < grid.ny) and (0 <= k < grid.nz)
    
    # Define Ghost Layer as valid coordinates that are NOT in the Core
    is_ghost = ((-1 <= i <= grid.nx) and (-1 <= j <= grid.ny) and (-1 <= k <= grid.nz)) and not is_core
    
    if is_core:
        cell = _build_core_cell(i, j, k, state)
    elif is_ghost:
        cell = _build_ghost_cell(i, j, k, state)
    else:
        # Defensive programming: Hard failure for any coordinate in the Padding Zone
        raise IndexError(f"[FACTORY] Out-of-bounds access: Requested cell ({i}, {j}, {k}) "
                         f"is in the illegal Padding Zone (Outside [-1, {grid.nx}]).")
    
    if DEBUG:
        print(f"DEBUG [Factory]: Allocated new {'Core' if is_core else 'Ghost'} "
              f"Cell at ({i}, {j}, {k}) | ID: {id(cell)}")
        
    return cell

def _build_core_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """Creates a View-based Core Cell."""
    grid = state.grid
    fields = state.fields
    init = state.initial_conditions
    mask_grid = state.mask.mask

    # Standardized: Core + 1 layer of ghosts on each side = +2
    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2

    # Shift core coordinates by +1 to skip the ghost padding
    index = get_flat_index(i + 1, j + 1, k + 1, nx_buf, ny_buf)
    
    cell = Cell(index=index, fields_buffer=fields.data, nx_buf=nx_buf, ny_buf=ny_buf, is_ghost=False)
    cell.vx, cell.vy, cell.vz = init.velocity
    cell.p = init.pressure
    cell.mask = int(mask_grid[i, j, k])
    
    return cell

def _build_ghost_cell(i: int, j: int, k: int, state: SolverState) -> Cell:
    """Creates a View-based Ghost Cell."""
    grid = state.grid
    nx_buf, ny_buf = grid.nx + 2, grid.ny + 2
    
    # Ghost cells already come in as -1 or N, so shifting them by +1 
    # maps them correctly to 0 and N+1 in the buffer
    index = get_flat_index(i + 1, j + 1, k + 1, nx_buf, ny_buf)
    
    cell = Cell(index=index, fields_buffer=state.fields.data, nx_buf=nx_buf, ny_buf=ny_buf, is_ghost=True)
    cell.vx, cell.vy, cell.vz = GHOST_VELOCITY
    cell.p = GHOST_PRESSURE
    cell.mask = GHOST_MASK
    
    return cell