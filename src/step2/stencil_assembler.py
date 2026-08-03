# src/step2/stencil_assembler.py

import logging

from src.common.field_schema import FI
from src.common.grid_math import get_flat_index
from src.common.solver_state import SolverState
from src.common.stencil_block import StencilBlock

from .factory import get_cell

# Rule 7: Granular Traceability
logger = logging.getLogger(__name__)

class CellRegistry:
    """
    Registry utilizing the Flyweight pattern to ensure unique Cell instances 
    across the grid, maintaining pointer integrity between adjacent stencils.
    """
    # Rule 0: Mandatory __slots__ to eliminate __dict__ overhead 
    # and minimize memory footprint.
    __slots__ = ['_cache', 'nx', 'nx_dim', 'ny', 'ny_dim', 'nz', 'nz_dim']

    def __init__(self, nx: int, ny: int, nz: int):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        # Total dimension: nx + 2 (includes ghost cells at -1 and nx)
        self.nx_dim = nx + 2
        self.ny_dim = ny + 2
        self.nz_dim = nz + 2
        # Use a standard Python list for objects to maintain pointer-integrity (Rule 0)
        self._cache = [None] * (self.nx_dim * self.ny_dim * self.nz_dim)

    def _get_idx(self, i: int, j: int, k: int) -> int:
        # Per Section 7: Valid coordinate range is [-1, nx]
        if not (-1 <= i <= self.nx and -1 <= j <= self.ny and -1 <= k <= self.nz):
             raise IndexError(f"Stencil accessing out-of-bounds: ({i}, {j}, {k})")
             
        # Explicit mapping: Shift [-1, N] to [0, N+1] for index calculation
        return get_flat_index(i + 1, j + 1, k + 1, self.nx_dim, self.ny_dim)

    def get_or_create(self, i: int, j: int, k: int, state: SolverState):
        idx = self._get_idx(i, j, k)
        if self._cache[idx] is None:
            self._cache[idx] = get_cell(i, j, k, state)
            logger.debug(f"Registry: Allocated new cell at ({i}, {j}, {k})")
        
        return self._cache[idx]

def assemble_stencil_matrix(state: SolverState) -> list:
    """
    Assembles a flattened list of StencilBlocks restricted to the Core Domain
    [0, nx-1] while maintaining access to Ghost buffers [-1, nx].
    
    This orchestration respects Rule 9 (Hybrid Memory Foundation) by using
    Python objects for logic-wiring while state data remains in NumPy.
    """
    if state.fields.data.shape[-1] != FI.num_fields():
        raise RuntimeError(f"Foundation Mismatch: Buffer width {state.fields.data.shape[-1]} "
                           f"!= Schema requirement {FI.num_fields()}.")

    grid = state.grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    logger.info(f"🚀 Stencil Assembly Started for {nx}x{ny}x{nz} Core Domain")
    
    registry = CellRegistry(nx, ny, nz)
    
    # Rule 5: Deterministic Initialization - Explicit extraction from SolverState
    physics_params = {
        "dx": grid.dx,
        "dy": grid.dy,
        "dz": grid.dz,
        "dt": state.simulation_parameters.time_step,
        "rho": state.fluid_properties.density,
        "mu": state.fluid_properties.viscosity,
        "f_vals": tuple(state.external_forces.force_vector)
    }

    local_stencil_list = []
    
    # Core loop: Iterate through logical core domain [0, N-1]
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                block = StencilBlock(
                    center=registry.get_or_create(i, j, k, state),
                    i_minus=registry.get_or_create(i - 1, j, k, state),
                    i_plus=registry.get_or_create(i + 1, j, k, state),
                    j_minus=registry.get_or_create(i, j - 1, k, state),
                    j_plus=registry.get_or_create(i, j + 1, k, state),
                    k_minus=registry.get_or_create(i, j, k - 1, state),
                    k_plus=registry.get_or_create(i, j, k + 1, state),
                    **physics_params
                )
                local_stencil_list.append(block)
    
    logger.info(f"✅ Successfully assembled {len(local_stencil_list)} Core StencilBlocks.")
    
    return local_stencil_list