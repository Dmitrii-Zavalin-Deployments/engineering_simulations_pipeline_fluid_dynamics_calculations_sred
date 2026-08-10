"""
src/archivist.py
Unified Archivist Module for snapshot serialization and final output packaging.
"""

import logging
import shutil
from pathlib import Path
import h5py
import numpy as np

from src.state import SolverState

logger = logging.getLogger("Solver.Archivist")


def save_snapshot(state: SolverState, base_dir: Path) -> Path:
    """
    Exports the physical 3D domain state to an HDF5 snapshot, stripping ghost cells [1:-1].
    """
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"snapshot_{state.current_iteration:04d}.h5"
    
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    x = np.linspace(state.grid.x_min, state.grid.x_max, nx)
    y = np.linspace(state.grid.y_min, state.grid.y_max, ny)
    z = np.linspace(state.grid.z_min, state.grid.z_max, nz)
    
    data = state.fields.data  # Contiguous Foundation buffer

    try:
        with h5py.File(filename, 'w') as h5f:
            def get_physical_3d(field_id: int):
                full_3d = data[:, field_id].reshape(nx + 2, ny + 2, nz + 2)
                return full_3d[1:-1, 1:-1, 1:-1]

            # Physical fields
            h5f.create_dataset("vx", data=get_physical_3d(0))
            h5f.create_dataset("vy", data=get_physical_3d(1))
            h5f.create_dataset("vz", data=get_physical_3d(2))
            h5f.create_dataset("p",  data=get_physical_3d(3))
            
            # Coordinates
            h5f.create_dataset('x', data=x)
            h5f.create_dataset('y', data=y)
            h5f.create_dataset('z', data=z)
            h5f.create_dataset('mask', data=state.mask)
            
            # Global metadata
            h5f.attrs['time'] = state.current_time
            h5f.attrs['iteration'] = state.current_iteration
            h5f.attrs['dx'] = (state.grid.x_max - state.grid.x_min) / (nx - 1 if nx > 1 else 1)
            h5f.attrs['dy'] = (state.grid.y_max - state.grid.y_min) / (ny - 1 if ny > 1 else 1)
            h5f.attrs['dz'] = (state.grid.z_max - state.grid.z_min) / (nz - 1 if nz > 1 else 1)
            
        state.saved_snapshots.append(str(filename))
        logger.info(f"ARCHIVIST [Success]: Snapshot iteration {state.current_iteration} saved to {filename}")
        return filename

    except Exception as e:
        logger.error(f"ARCHIVIST [Critical Failure]: Could not write {filename} | Error: {e!s}")
        raise


def archive_simulation_artifacts(state: SolverState, base_dir: Path, output_file_path: Path) -> str:
    """
    Instance-Optimized Archiver.
    Packages raw output snapshots and results into a structured ZIP archive for SSoT storage.
    """
    source_dir = (base_dir / "output").resolve()
    target_dir = output_file_path.parent
    staging_dir = Path.cwd() / "navier_stokes_output"
    
    logger.info(f"ARCHIVE: Initiating artifact packaging from {source_dir}")

    if not source_dir.exists():
        logger.critical(f"ARCHIVE FAILED: Source directory missing at {source_dir}")
        raise FileNotFoundError(f"Source directory '{source_dir}' not found.")

    target_dir.mkdir(parents=True, exist_ok=True)

    if staging_dir.exists():
        logger.warning(f"ARCHIVE: Existing staging folder detected at {staging_dir}. Clearing...")
        shutil.rmtree(staging_dir)
    
    shutil.move(str(source_dir), str(staging_dir))
    logger.info(f"ARCHIVE: Source moved to staging: {staging_dir}")

    temp_zip_path = shutil.make_archive(str(staging_dir), 'zip', str(staging_dir))
    logger.info(f"ARCHIVE: ZIP package created at {temp_zip_path}")

    final_destination = target_dir / "navier_stokes_output.zip"
    if final_destination.exists():
        logger.debug(f"ARCHIVE: Overwriting existing archive at {final_destination}")
        final_destination.unlink()

    shutil.move(temp_zip_path, str(final_destination))
    logger.info(f"ARCHIVE COMPLETE: Artifacts anchored to {final_destination}")

    return str(final_destination)
