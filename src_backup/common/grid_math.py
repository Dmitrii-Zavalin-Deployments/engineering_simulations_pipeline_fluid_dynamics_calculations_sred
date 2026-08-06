# src/common/grid_math.py

def get_flat_index(i: int, j: int, k: int, nx: int, ny: int) -> int:
    """
    Computes a flat index from 3D coordinates. 
    Assumes standard row-major order: index = i + nx * j + (nx * ny) * k
    """
    return i + (nx * j) + (nx * ny * k)

def get_coords_from_index(index: int, nx: int, ny: int) -> tuple[int, int, int]:
    """
    SSoT Mapping: Converts flat index back to (i, j, k).
    """
    xy_plane = nx * ny
    
    k = index // xy_plane
    rem = index % xy_plane
    j = rem // nx
    i = rem % nx
    
    return i, j, k