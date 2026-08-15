"""
src/archivist.py
Archivist Module.
Serializes final field states (u, v, w, p), packages snapshot binaries into a timestamped
ZIP archive, and generates the canonical output JSON manifest adhering to navier_stokes_output.schema.json.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

logger = logging.getLogger("Solver.Archivist")


def archive_simulation_results(
    state: Any,
    output_dir: str | Path,
    output_json_filename: str = "navier_stokes_output.json",
    zip_filename: str | None = None,
) -> str:
    """
    Exports solved fields to NPY snapshots, creates a timestamped ZIP archive,
    and saves the schema-compliant output JSON file.

    Args:
        state: Sovereign SolverState container holding final simulation state.
        output_dir: Target directory path for output artifacts.
        output_json_filename: File name for the output JSON manifest.
        zip_filename: Optional name for the ZIP archive. If None, generates YYYYMMDD_HHMMSS.zip.

    Returns:
        Absolute string path to the generated output JSON manifest file.
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided (no defaults allowed).")
    if output_dir is None:
        raise ValueError("FATAL ERROR: output_dir must be explicitly provided (no defaults allowed).")

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Determine ZIP archive filename (timestamped if not specified)
    if not zip_filename or zip_filename.endswith(".json"):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    # 4. Construct Schema-Compliant Output JSON Payload
    output_payload: dict[str, Any] = {
        "inputs": state.input_data,
        "config": state.config,
        "results": {
            "status": "SUCCESS",
            "zip_filename": target_zip_name,
        },
    }

    # 5. Write output JSON manifest
    json_path = out_path / output_json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info(f"Successfully written output JSON manifest to: {json_path}")
    return str(json_path)
