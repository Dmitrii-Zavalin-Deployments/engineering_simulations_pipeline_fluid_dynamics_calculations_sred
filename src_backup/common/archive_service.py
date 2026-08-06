# src/common/archive_service.py

import logging
import shutil
from pathlib import Path

from src.common.solver_state import SolverState

# Rule 5: Explicit Forensic Logging
logger = logging.getLogger(__name__)

def archive_simulation_artifacts(state: SolverState) -> str:
    """
    Instance-Optimized Archiver.
    Moves raw output to a structured staging folder and packages for SSoT storage.
    Note: Cleanup is intentionally omitted as instances are ephemeral (Rule 8).
    """
    # 1. Resolve Dynamic Paths via SSoT
    # We pull the live BASE_DIR to ensure consistency across environments.
    import src.main
    current_base = Path(src.main.BASE_DIR)

    # Source: Where the solver results currently reside
    source_dir = Path(state.manifest.output_directory).resolve()
    
    # Target: Final destination in the project data repository
    target_dir = current_base / "data" / "testing-input-output"
    
    # Staging: Temporary folder in CWD to preserve ZIP internal structure
    staging_dir = Path.cwd() / "navier_stokes_output"
    
    logger.info(f"ARCHIVE: Initiating artifact packaging from {source_dir}")

    # 2. Safety Check (Rule 5: Explicit or Error)
    if not source_dir.exists():
        logger.critical(f"ARCHIVE FAILED: Source directory missing at {source_dir}")
        raise FileNotFoundError(f"Source directory '{source_dir}' not found.")

    # 3. Ensure Infrastructure exists
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"ARCHIVE: Target infrastructure verified at {target_dir}")

    # 4. Atomic Staging
    # If a previous local run left a folder, we clear it to prevent "Frankenstein" merges.
    if staging_dir.exists():
        logger.warning(f"ARCHIVE: Existing staging folder detected at {staging_dir}. Clearing...")
        shutil.rmtree(staging_dir)
    
    shutil.move(str(source_dir), str(staging_dir))
    logger.info(f"ARCHIVE: Source moved to staging: {staging_dir}")

    # 5. Package into Archive
    # Note: Using staging_dir as base ensures the ZIP doesn't contain the full system path.
    temp_zip_path = shutil.make_archive(str(staging_dir), 'zip', str(staging_dir))
    logger.info(f"ARCHIVE: ZIP package created at {temp_zip_path}")

    # 6. Final Placement
    final_destination = target_dir / "navier_stokes_output.zip"
    if final_destination.exists():
        logger.debug(f"ARCHIVE: Overwriting existing archive at {final_destination}")
        final_destination.unlink()

    shutil.move(temp_zip_path, str(final_destination))
    
    # 7. Lifecycle Log
    # No shutil.rmtree(staging_dir) here; handled by GHA instance dissolution.
    logger.info(f"ARCHIVE COMPLETE: Artifacts anchored to {final_destination}")

    return str(final_destination)