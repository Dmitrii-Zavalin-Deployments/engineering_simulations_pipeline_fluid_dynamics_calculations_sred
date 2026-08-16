"""
src/archivist.py
Archivist Module.
Serializes intermediate and final field states (u, v, w, p) using zero-padded step indexing, 
packages all snapshot binaries into a timestamped ZIP archive on success, cleans up loose temporary files, 
and generates canonical output JSON manifests adhering strictly to navier_stokes_output.schema.json.
"""

import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("Solver.Archivist")


def export_step_snapshot(
    state: Any,
    step: int,
    output_dir: str | Path,
) -> list[Path]:
    """
    Exports 3D field snapshots (u, v, w, p) retaining spatial dimensions (nx, ny, nz)
    tagged by zero-padded step index.

    Args:
        state: Sovereign SolverState container holding simulation state.
        step: Current simulation iteration index.
        output_dir: Target directory path for output artifacts.

    Returns:
        List of Path objects for exported .npy files.
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided (no defaults allowed).")
    if output_dir is None:
        raise ValueError("FATAL ERROR: output_dir must be explicitly provided (no defaults allowed).")

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

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

    step_str = f"{step:06d}"
    field_names = ["field_u", "field_v", "field_w", "field_p"]
    saved_snapshots: list[Path] = []

    for idx, name in enumerate(field_names):
        if idx < len(fields):
            field_data = fields[idx]
            npy_path = out_path / f"{name}_step_{step_str}.npy"
            np.save(npy_path, field_data)
            saved_snapshots.append(npy_path)
            logger.info(f"Exported field snapshot: {npy_path.name} (Shape: {field_data.shape})")

    return saved_snapshots


def archive_simulation_results(
    state: Any,
    output_dir: str | Path,
    output_filename: str,
    status: str,
) -> None:
    """
    Exports final fields, packages all step snapshot artifacts into a timestamped ZIP archive,
    cleans up temporary binaries, and builds schema-compliant JSON manifest.

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

        # 1. Export final state snapshot to guarantee final frame presence if not already dumped
        final_step = getattr(state, "current_iteration", 0)
        final_snapshot_check = out_path / f"field_u_step_{final_step:06d}.npy"
        if not final_snapshot_check.exists():
            export_step_snapshot(
                state=state,
                step=final_step,
                output_dir=out_path,
            )

        # 2. Gather all generated step snapshot binaries
        saved_snapshots = sorted(out_path.glob("field_*_step_*.npy"))

        # 3. Generate UTC timestamped ZIP archive filename (YYYYMMDD_HHMMSS.zip)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target_zip_name = f"{timestamp_str}.zip"

        # 4. Compress all snapshot binaries into timestamped ZIP archive
        zip_file_path = out_path / target_zip_name
        logger.info(f"Creating output ZIP archive: {zip_file_path}")

        with zipfile.ZipFile(zip_file_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_out:
            for npy_file in saved_snapshots:
                zip_out.write(npy_file, arcname=npy_file.name)

        logger.info(f"Successfully archived {len(saved_snapshots)} snapshot binaries into: {zip_file_path.name}")

        # 5. Clean up loose temporary .npy files so only the ZIP archive remains
        for npy_file in saved_snapshots:
            try:
                npy_file.unlink()
                logger.info(f"Cleaned up temporary uncompressed snapshot: {npy_file.name}")
            except OSError as e:
                logger.warning(f"Failed to delete temporary npy file {npy_file.name}: {e}")
    else:
        target_zip_name = "NOT_APPLICABLE"
        logger.warning("Simulation marked as FAILURE. Skipping snapshot binary creation.")

    # 6. Construct Schema-Compliant Output JSON Payload matching navier_stokes_output.schema.json
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
