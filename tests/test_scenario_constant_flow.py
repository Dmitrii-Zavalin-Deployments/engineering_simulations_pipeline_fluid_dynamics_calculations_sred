"""
Literate Integration Test: Scenario 1 - Constant Flow Field Integration.
Validates the Navier-Stokes execution engine for advection-dominated steady inflow
across all main pipeline stages without body force acceleration.
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# Module under test
from src.main import main, run_simulation


def test_integration_constant_flow_pipeline(workspace_folder, monkeypatch):
    # In this scenario, we evaluate steady physical transport where initial velocities
    # match the boundary inflow conditions, and external body forces are set to zero.
    # The computational grid is configured to 4x4x4 (64 cells total).
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "constant_flow_manifest.json"

    # We read the base configuration to customize it for zero external force.
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Dimensional configuration:
    #     nx = 4, ny = 4, nz = 4 -> Total grid cells N = 4 * 4 * 4 = 64
    input_data["grid"].update({"nx": 4, "ny": 4, "nz": 4})
    input_data["mask"] = [0] * 64

    # Initial velocities and boundary values are matched:
    #     u_0 = 0.0, v_0 = 0.0, w_0 = 1.0 m/s
    input_data["initial_conditions"]["velocity"] = [0.0, 0.0, 1.0]
    input_data["external_forces"]["force_vector"] = [0.0, 0.0, 0.0]
    input_data["external_forces"]["gravity_vector"] = [0.0, 0.0, 0.0]

    # Inflow/Outflow boundary configuration:
    input_data["boundary_conditions"] = [
        {"location": "z_min", "type": "inflow", "values": {"u": 0.0, "v": 0.0, "w": 1.0, "p": 0.0}},
        {"location": "z_max", "type": "outflow", "values": {"u": 0.0, "v": 0.0, "w": 1.0, "p": 0.0}},
    ]

    # Write input configuration back to disk for pipeline ingestion verification
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2)

    # Configure command-line execution arguments for src/main.py
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_manifest_name,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # Execute full pipeline through entrypoint (Ingestion -> State -> C++ Engine -> Archivist)
    main()

    # Stage 1 Verification: Pipeline Output Manifest Generation
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Output JSON manifest not written to {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify root level schema structural compliance
    assert "inputs" in manifest
    assert "config" in manifest
    assert "results" in manifest

    # Stage 2 Verification: Ingestion Parameter Parity
    inputs_block = manifest["inputs"]
    assert inputs_block["grid"]["nx"] == 4
    assert inputs_block["grid"]["ny"] == 4
    assert inputs_block["grid"]["nz"] == 4
    assert inputs_block["external_forces"]["force_vector"] == [0.0, 0.0, 0.0]

    # Stage 3 Verification: Execution State and ZIP Container Creation
    results_block = manifest["results"]
    assert results_block["status"] == "SUCCESS"

    zip_filename = results_block.get("zip_filename")
    assert zip_filename and zip_filename != "NOT_APPLICABLE"
    
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"Archived simulation results file missing at {zip_path}"

    # Stage 4 Verification: Binary Output Array Inspection (.npy)
    # Total iterations executed based on step count:
    #     final_step = total_time / time_step = 0.003 / 0.001 = 3
    final_step = 3
    expected_files = [
        f"field_u_step_{final_step:06d}.npy",
        f"field_v_step_{final_step:06d}.npy",
        f"field_w_step_{final_step:06d}.npy",
        f"field_p_step_{final_step:06d}.npy",
    ]

    with zipfile.ZipFile(zip_path, "r") as zf:
        archive_files = zf.namelist()
        for expected_file in expected_files:
            assert expected_file in archive_files, f"Expected output binary snapshot {expected_file} missing."

            # Load field data from zip stream
            raw_bytes = zf.read(expected_file)
            field_arr = np.load(io.BytesIO(raw_bytes))

            # Shape verification: 4x4x4 spatial array
            assert field_arr.shape == (4, 4, 4)

            # Numerical integrity verification: No NaN or Infinity propagation
            assert not np.isnan(field_arr).any(), f"NaN values encountered in {expected_file}"
            assert not np.isinf(field_arr).any(), f"Inf values encountered in {expected_file}"

            # Constant Z-velocity check: w should maintain ~1.0 m/s transport velocity
            if "field_w" in expected_file:
                assert np.allclose(field_arr, 1.0, atol=1e-5), f"Z-velocity drifted from nominal 1.0 value in {expected_file}"
