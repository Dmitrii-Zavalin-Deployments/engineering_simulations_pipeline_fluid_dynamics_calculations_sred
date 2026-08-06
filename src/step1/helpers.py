import logging
import numpy as np

from src.common.solver_input import GridInput

logger = logging.getLogger(__name__)

# Rule 7: Granular Traceability
DEBUG = False


def generate_3d_masks(mask_data: list[int], grid: GridInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms flat input into 3D topology arrays via SSoT vectorized NumPy operations.
    
    Compliance:
    - Rule 4 (SSoT): Mapping logic derived from grid_math.py indexing order.
    - Rule 5 (Deterministic): Strict boundary validation; no implicit resizing.
    """
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    
    # 1. Strict Size Integrity Check
    # Ensure the input flat list perfectly matches the 3D grid volume.
    expected_size = nx * ny * nz
    if len(mask_data) != expected_size:
        raise ValueError(f"Mask data size mismatch: Expected {expected_size} cells, got {len(mask_data)}")
    
    # 2. Vectorized mapping via Fortran-ordered reshape (eliminating Python scalar loop overhead)
    mask_array = np.array(mask_data, dtype=np.int8)
    mask_3d = mask_array.reshape((nx, ny, nz), order='F')

    # 3. Logic-Layer: Identify fluid and boundary regions via vectorized masks
    is_fluid = (mask_3d == 1)
    is_boundary = (mask_3d == -1)
    
    if DEBUG:
        logger.debug("DEBUG [Step 1.2]: Topology Verification (Mask Generated)")
        logger.debug(f"  > Grid Dimensions: {nx}x{ny}x{nz}")
        logger.debug(f"  > Fluid Volume: {np.sum(is_fluid)} cells")
        
    return mask_3d, is_fluid, is_boundary
