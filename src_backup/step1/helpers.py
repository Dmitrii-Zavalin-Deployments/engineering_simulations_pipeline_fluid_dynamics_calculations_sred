# src/step1/helpers.py

import numpy as np

from src.common.grid_math import get_coords_from_index
from src.common.solver_input import GridInput

# Rule 7: Granular Traceability
DEBUG = False


def generate_3d_masks(mask_data: list[int], grid: GridInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms flat input into 3D topology arrays via explicit SSoT mapping.
    
    Compliance:
    - Rule 4 (SSoT): Mapping logic derived from grid_math.py.
    - Rule 5 (Deterministic): Strict boundary validation; no implicit resizing.
    """
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    
    # 1. Strict Size Integrity Check
    # Ensure the input flat list perfectly matches the 3D grid volume.
    expected_size = nx * ny * nz
    if len(mask_data) != expected_size:
        raise ValueError(f"Mask data size mismatch: Expected {expected_size} cells, got {len(mask_data)}")
    
    # 2. Initialize empty buffer
    mask_3d = np.zeros((nx, ny, nz), dtype=np.int8)
    
    # 3. Explicit mapping (Eliminating "Black Box" reshape)
    # We map the flat input list to the 3D grid using our SSoT logic.
    for idx, value in enumerate(mask_data):
        i, j, k = get_coords_from_index(idx, nx, ny) 
        
        # Guard against index drift (Deterministic validation)
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            mask_3d[i, j, k] = value
        else:
            raise ValueError(f"Mask mapping overflow at index {idx} -> ({i}, {j}, {k})")

    # 4. Logic-Layer: Identify fluid and boundary regions via vectorized masks
    is_fluid = (mask_3d == 1)
    is_boundary = (mask_3d == -1)
    
    if DEBUG:
        print("DEBUG [Step 1.2]: Topology Verification (Mask Generated)")
        print(f"  > Grid Dimensions: {nx}x{ny}x{nz}")
        print(f"  > Fluid Volume: {np.sum(is_fluid)} cells")
        
    return mask_3d, is_fluid, is_boundary