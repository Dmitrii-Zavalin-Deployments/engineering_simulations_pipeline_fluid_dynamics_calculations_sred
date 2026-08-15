"""
src/archivist.py
Archivist Module.
Serializes final field states (u, v, w, p), packages snapshot binaries into a timestamped
ZIP archive on success, and generates canonical output JSON manifests adhering to 
navier_stokes_output.schema.json for both success and failure states.
"""

import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("Solver.Archivist")


def archive_simulation_results(
    state: Any,
    output_dir: str | Path,
    output_filename: str = "simulation_results.zip",
    status: str = "SUCCESS",
) -> None:
    """
    Exports solved fields, builds schema-compliant JSON manifest, and packages artifacts.

    Args:
        state: Sovereign SolverState container holding simulation state.
        output_dir: Target directory path for output artifacts.
        output_filename: File name for the output ZIP archive or output JSON manifest.
        status: Execution status string ("SUCCESS" or "FAILURE").
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided (no defaults allowed).")
    if output_dir is None:
        raise ValueError("FATAL ERROR: output_dir must be explicitly provided (no defaults allowed).")

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    normalized_status = status.upper()

    if output_filename.endswith(".json"):
        output_json_filename = output_filename
        zip_filename = None
    else:
        output_json_filename = "navier_stokes_output.json"
        zip_filename = output_filename

    if normalized_status == "SUCCESS":
        # 1. Determine ZIP archive filename (timestamped UTC if zip_filename is None or ends with .json)
        if not zip_filename or zip_filename.endswith(".json"):
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            target_zip_name = f"{timestamp_str}.zip"
        else:
            target_zip_name = zip_filename if zip_filename.endswith(".zip") else f"{zip_filename}.zip"

        # 2. Export 1D field snapshots (u, v, w, p) matching C-order row-major indexing
        saved_snapshots: list[Path] = []
        field_names = ["field_u", "field_v", "field_w", "field_p"]

        for idx, name in enumerate(field_names):
            field_1d = state.fields[idx].ravel(order="C")
            npy_path = out_path / f"{name}.npy"
            np.save(npy_path, field_1d)
            saved_snapshots.append(npy_path)
            logger.info(f"Exported field snapshot: {npy_path.name} (Length: {len(field_1d)})")

        # 3. Compress snapshot binaries into ZIP archive
        zip_file_path = out_path / target_zip_name
        logger.info(f"Creating output ZIP archive: {zip_file_path}")

        with zipfile.ZipFile(zip_file_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_out:
            for npy_file in saved_snapshots:
                zip_out.write(npy_file, arcname=npy_file.name)

        logger.info(f"Successfully archived snapshot binaries into: {zip_file_path.name}")
    else:
        target_zip_name = "NOT_APPLICABLE"
        logger.warning("Simulation marked as FAILURE. Skipping snapshot binary creation.")

    # 4. Construct Schema-Compliant Output JSON Payload
    output_payload: dict[str, Any] = {
        "inputs": getattr(state, "input_data", {}),
        "config": getattr(state, "config", {}),
        "results": {
            "status": normalized_status,
            "zip_filename": target_zip_name,
        },
    }

    # 5. Write output JSON manifest
    json_path = out_path / output_json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info(f"Successfully written output JSON manifest to: {json_path}")
