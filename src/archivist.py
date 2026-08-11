"""
src/archivist.py
Output Packaging & Archival Module.
Compiles simulation statistics, writes metadata JSON files, and packages outputs into ZIP archives.
"""

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("Solver.Archivist")


def archive_simulation_results(state, output_dir: str, zip_filename: str = "simulation_results.zip") -> str:
    """
    Compiles final simulation state, writes output JSON metadata, and packages
    all generated previews and configuration files into a compressed ZIP archive.

    Args:
        state: SolverState instance after execution finishes
        output_dir: Directory containing output files and previews
        zip_filename: Name of the generated zip archive

    Returns:
        Absolute path to the created ZIP archive.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_metadata: dict[str, Any] = {
        "status": "COMPLETED",
        "total_iterations_completed": state.current_iteration,
        "final_simulation_time": state.current_time,
        "target_total_time": state.total_time,
        "grid_dimensions": {
            "nx": state.nx,
            "ny": state.ny,
            "nz": state.nz,
        },
        "fluid_properties": state.fluid_properties,
        "snapshots": state.snapshot_records,
    }

    # Write summary metadata JSON
    summary_file = out_path / "output_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_metadata, f, indent=2)

    logger.info(f"Written simulation summary metadata: {summary_file}")

    # Package files into ZIP archive
    zip_path = out_path / zip_filename
    logger.info(f"Creating output archive: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_out:
        # Add summary json
        zip_out.write(summary_file, arcname="output_summary.json")

        # Add all previews present in output directory
        for item in out_path.glob("preview_step_*.*"):
            zip_out.write(item, arcname=f"previews/{item.name}")

    logger.info(f"Archival complete. Output stored at {zip_path}")
    return str(zip_path.resolve())
