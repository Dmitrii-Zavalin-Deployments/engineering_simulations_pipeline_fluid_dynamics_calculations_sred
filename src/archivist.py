# src/step4/io_archivist.py

import logging
from pathlib import Path

import h5py
import numpy as np

from src.common.field_schema import FI

# Rule 7: Granular Traceability for IO Finalization
logger = logging.getLogger("Solver.Step4.Archivist")

def save_snapshot(state) -> None:
    """
    Exports the physical 3D domain state to HDF5, stripping ghost cells.
    
    Compliance:
    - Rule 4 (SSoT): Accesses grid and state data from authorized sub-containers.
    - Rule 8 (Law of Singular Access): Coordinates computed locally to avoid 'God Object' properties.
    - Rule 9 (Hybrid Memory): Direct Foundation slicing via [1:-1] Ghost Stripping.
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Rule 5: Explicit derivation of state properties (Filename SSoT)
    filename = output_dir / f"snapshot_{state.iteration:04d}.h5"
    
    # Retrieve physical dimensions (N) from the authorized Grid container
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    # Compute physical coordinate meshes (N) on-the-fly (Rule 8 compliance)
    x = np.linspace(state.grid.x_min, state.grid.x_max, nx)
    y = np.linspace(state.grid.y_min, state.grid.y_max, ny)
    z = np.linspace(state.grid.z_min, state.grid.z_max, nz)
    
    # Access the contiguous Foundation buffer (N+2) via FieldManager (Rule 9)
    # The 'data' buffer is a 1D flat array of shape ( (nx+2)*(ny+2)*(nz+2), num_fields )
    data = state.fields.data 

    try:
        with h5py.File(filename, 'w') as h5f:
            # Helper for clean slicing: Reshape flat buffer to 3D and strip ghost layers [1:-1]
            # This ensures HDF5[0,0,0] matches the first physical cell (not the ghost cell).
            def get_physical_3d(field_id: FI):
                full_3d = data[:, field_id].reshape(nx+2, ny+2, nz+2)
                return full_3d[1:-1, 1:-1, 1:-1]

            # Physical Fields: Direct, schema-locked slicing and ghost stripping
            h5f.create_dataset("vx", data=get_physical_3d(FI.VX))
            h5f.create_dataset("vy", data=get_physical_3d(FI.VY))
            h5f.create_dataset("vz", data=get_physical_3d(FI.VZ))
            h5f.create_dataset("p",  data=get_physical_3d(FI.P))
            
            # Spatial metadata: Using computed physical arrays (N size)
            h5f.create_dataset('x', data=x)
            h5f.create_dataset('y', data=y)
            h5f.create_dataset('z', data=z)
            
            # Grid mask: Stripped of ghost cells to match field shapes
            # If state.mask.mask is already physical size, no slicing is needed.
            # Otherwise, apply [1:-1, 1:-1, 1:-1] as per the field logic.
            h5f.create_dataset('mask', data=state.mask.mask)
            
            # Global Metadata: Explicit attribution for solver state traceability
            h5f.attrs['time'] = state.time
            h5f.attrs['iteration'] = state.iteration
            h5f.attrs['dx'] = (state.grid.x_max - state.grid.x_min) / (nx - 1 if nx > 1 else 1)
            h5f.attrs['dy'] = (state.grid.y_max - state.grid.y_min) / (ny - 1 if ny > 1 else 1)
            h5f.attrs['dz'] = (state.grid.z_max - state.grid.z_min) / (nz - 1 if nz > 1 else 1)
            
        # Update manifest via the state object for archival tracking
        state.manifest.saved_snapshots.append(str(filename))
        logger.info(f"ARCHIVIST [Success]: Snapshot {state.iteration} saved to {filename}")

    except Exception as e:
        logger.error(f"ARCHIVIST [Critical Failure]: Could not write {filename} | Error: {e!s}")
        raise