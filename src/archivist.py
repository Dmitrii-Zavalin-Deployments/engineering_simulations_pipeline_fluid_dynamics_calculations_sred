"""
src/archivist.py
Archivist Module.
Serializes final field states (u, v, w, p), packages snapshot binaries into a timestamped
ZIP archive on success, and generates canonical output JSON manifests adhering strictly
to navier_stokes_output.schema.json for both success and failure states.
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
    output_filename: str,
    status: str,
) -> None:
    """
    Exports solved fields, builds schema-compliant JSON manifest, and packages artifacts.

    Args:
        state: Sovereign SolverState container holding simulation state.
        output_dir: Target directory path for output artifacts.
        output_filename: File name for the output JSON manifest (e.g. navier_stokes_output.json).
        status: Execution status string ("SUCCESS" or "FAILURE").
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided (no defaults allowed).")
    if output_dir is None:
        raise ValueError("FATAL ERROR: output_dir must be explicitly provided (no defaults allowed).")
    if output_filename is None:
        raise ValueError("FATAL ERROR: output_filename must be explicitly provided (no defaults allowed).")
    if status is None:
        raise ValueError("FATAL ERROR: status must be explicitly provided (no defaults allowed).")

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    normalized_status = status.upper()

    if normalized_status == "SUCCESS":
        # 0. Sync C++ solver fields back to state if bound
        if hasattr(state, "_cpp_solver") and state._cpp_solver is not None:
            if hasattr(state._cpp_solver, "sync_fields") and callable(state._cpp_solver.sync_fields):
                state._cpp_solver.sync_fields(state)
            elif hasattr(state._cpp_solver, "get_fields") and callable(state._cpp_solver.get_fields):
                updated_fields = state._cpp_solver.get_fields()
                if updated_fields is not None:
                    state.fields = np.array(updated_fields, copy=True)
            else:
                raise RuntimeError(
                    "FATAL ERROR: C++ solver instance attached to state is missing required 'sync_fields' or 'get_fields' binding."
                )

        # 1. Generate UTC timestamped ZIP archive filename (YYYYMMDD_HHMMSS.zip)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target_zip_name = f"{timestamp_str}.zip"

        # 2. Export 3D field snapshots (u, v, w, p) retaining spatial dimensions (nx, ny, nz)
        saved_snapshots: list[Path] = []
        field_names = ["field_u", "field_v", "field_w", "field_p"]

        fields = getattr(state, "fields", None)
        if fields is None:
            u_f = getattr(state, "u", None)
            v_f = getattr(state, "v", None)
            w_f = getattr(state, "w", None)
            p_f = getattr(state, "p", None)
            if u_f is not None and v_f is not None and w_f is not None and p_f is not None:
                fields = np.stack([u_f, v_f, w_f, p_f], axis=0)

        if fields is None:
            raise ValueError("FATAL ERROR: state.fields must be explicitly provided and populated (no defaults allowed).")

        for idx, name in enumerate(field_names):
            if idx < len(fields):
                field_data = fields[idx]
                npy_path = out_path / f"{name}.npy"
                np.save(npy_path, field_data)
                saved_snapshots.append(npy_path)
                logger.info(f"Exported field snapshot: {npy_path.name} (Shape: {field_data.shape})")

        # 3. Compress snapshot binaries into timestamped ZIP archive
        zip_file_path = out_path / target_zip_name
        logger.info(f"Creating output ZIP archive: {zip_file_path}")

        with zipfile.ZipFile(zip_file_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_out:
            for npy_file in saved_snapshots:
                zip_out.write(npy_file, arcname=npy_file.name)

        logger.info(f"Successfully archived snapshot binaries into: {zip_file_path.name}")
    else:
        target_zip_name = "NOT_APPLICABLE"
        logger.warning("Simulation marked as FAILURE. Skipping snapshot binary creation.")

    # 4. Construct Schema-Compliant Output JSON Payload matching navier_stokes_output.schema.json
    config_obj = getattr(state, "config", getattr(state, "config_data", {}))
    output_payload: dict[str, Any] = {
        "inputs": getattr(state, "input_data", {}),
        "config": config_obj,
        "results": {
            "status": normalized_status,
            "zip_filename": target_zip_name,
        },
    }

    json_file_path = out_path / output_filename
    logger.info(f"Writing output JSON manifest to: {json_file_path}")
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    logger.info("Successfully wrote output JSON manifest")
