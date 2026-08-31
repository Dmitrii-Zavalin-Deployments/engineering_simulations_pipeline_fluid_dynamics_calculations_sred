"""
Literate Integration Test: Scenario 3 - Accelerated Flow with Gravity Integration.
Validates the Navier-Stokes execution engine under combined driving force vectors
and vertical downward gravity acceleration (g_y = -9.81 m/s^2).
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# Module under test
from src.main import main


def test_integration_accelerated_gravity_pipeline(workspace_folder, monkeypatch):
    # In this scenario, fluid dynamics are subjected to both horizontal body forces
    # and strong vertical gravity acceleration.
    # Theoretical gravity force vector:
    #     g = [0.0, -9.81, 0.0] m/s^2
    # Body force vector:
    #     F = [1.0, 0.0, 0.0] N/kg
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "accelerated_gravity_manifest.json"

    # Ingest baseline payload
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Set up asymmetrical non-cubic spatial resolution 5x4x3 (60 cells total)
    # to test stride and memory allocation under multi-directional forcing
    input_data["grid"].update({"nx": 5, "ny": 4, "nz": 3})
    input_data["mask"] = [0] * 60

    # Initial condition setup
    input_data["initial_conditions"]["velocity"] = [0.1, 0.0, 0.0]

    # Forces setup: Horizontal acceleration + vertical gravity
    input_data["external_forces"]["force_vector"] = [1.0, 0.0, 0.0]
    input_data["external_forces"]["gravity_vector"] = [0.0, -9.81, 0.0]

    # Save test configuration
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2)

    # Setup command-line environment flags
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_manifest_name,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # Run complete unmocked pipeline
    main()

    # Stage 1 Verification: Manifest Output File Existence
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Output JSON manifest missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Stage 2 Verification: Gravity Vector Ingestion Parity
    assert manifest["inputs"]["external_forces"]["gravity_vector"] == [0.0, -9.81, 0.0]

    # Stage 3 Verification: Execution Success and ZIP Packaging
    assert manifest["results"]["status"] == "SUCCESS"
    zip_filename = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"Output ZIP file missing at {zip_path}"

    # Stage 4 Verification: Binary Array Shape and Negative Gravity Field Response
    final_step = 3
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Check v-velocity component file for negative downward acceleration
        v_snapshot_name = f"field_v_step_{final_step:06d}.npy"
        assert v_snapshot_name in zf.namelist(), f"Snapshot {v_snapshot_name} missing from ZIP archive."

        array_bytes = zf.read(v_snapshot_name)
        v_field = np.load(io.BytesIO(array_bytes))

        # Shape verification for asymmetric 5x4x3 grid
        assert v_field.shape == (5, 4, 3)

        # Numerical bounds check
        assert not np.isnan(v_field).any(), f"NaN values detected in {v_snapshot_name}"
        assert not np.isinf(v_field).any(), f"Inf values detected in {v_snapshot_name}"

        # Physical response check: Downward gravity must produce non-zero negative vertical velocity field
        assert np.min(v_field) < 0.0, (
            f"Vertical velocity field failed to respond to downward gravity vector g_y = -9.81. Min value: {np.min(v_field)}"
        )
