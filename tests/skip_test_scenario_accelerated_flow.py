"""
Literate Integration Test: Scenario 2 - Accelerated Flow Field Integration.
Validates the Navier-Stokes execution engine under constant body force acceleration
vector F = [1.0, 1.0, 1.0] across all main pipeline execution stages.
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

# Module under test
from src.main import main


def test_integration_accelerated_flow_pipeline(workspace_folder, monkeypatch):
    # Under constant body force acceleration, momentum is injected into all 
    # spatial dimensions. For force vector F = [1.0, 1.0, 1.0], velocity components 
    # must monotonically mutate across solver iterations.
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "accelerated_flow_manifest.json"

    # Ingest baseline input payload to configure spatial grid and forces
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Spatial setup for 4x4x4 domain (64 cells total)
    input_data["grid"].update({"nx": 4, "ny": 4, "nz": 4})
    input_data["mask"] = [0] * 64

    # Set initial velocity state: u_0 = 0.1, v_0 = 0.1, w_0 = 0.1 m/s
    input_data["initial_conditions"]["velocity"] = [0.1, 0.1, 0.1]

    # Accelerating body forces:
    #     fx = 1.0 N/kg, fy = 1.0 N/kg, fz = 1.0 N/kg
    input_data["external_forces"]["force_vector"] = [1.0, 1.0, 1.0]
    input_data["external_forces"]["gravity_vector"] = [0.0, 0.0, 0.0]

    # Write modified payload to disk
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2)

    # Configure CLI invocation arguments
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_manifest_name,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # Execute full pipeline through entrypoint
    main()

    # Stage 1 Verification: Output Manifest File Existence
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Output JSON manifest not created at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Stage 2 Verification: Force Vector Ingestion Compliance
    assert manifest["inputs"]["external_forces"]["force_vector"] == [1.0, 1.0, 1.0]

    # Stage 3 Verification: Solver Results and ZIP Container Integrity
    assert manifest["results"]["status"] == "SUCCESS"
    zip_filename = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"ZIP archive missing at {zip_path}"

    # Stage 4 Verification: C++ Field Mutation Inspection
    # Validate non-zero velocity field development driven by body force acceleration
    final_step = 3
    field_names = ["field_u", "field_v", "field_w", "field_p"]

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in field_names:
            snapshot_filename = f"{name}_step_{final_step:06d}.npy"
            assert snapshot_filename in zf.namelist(), f"Snapshot {snapshot_filename} missing from archive."

            raw_bytes = zf.read(snapshot_filename)
            field_data = np.load(io.BytesIO(raw_bytes))

            # Shape checking for 4x4x4 domain
            assert field_data.shape == (4, 4, 4)

            # Ensure stability: No NaN or Inf numerical explosions
            assert not np.isnan(field_data).any(), f"NaN detected in {snapshot_filename}"
            assert not np.isinf(field_data).any(), f"Inf detected in {snapshot_filename}"

            # Acceleration verification: Field magnitude must exceed initial conditions (> 0.1)
            if name in ["field_u", "field_v", "field_w"]:
                assert np.max(np.abs(field_data)) > 0.1, (
                    f"Velocity field {name} failed to accelerate under constant force."
                )
