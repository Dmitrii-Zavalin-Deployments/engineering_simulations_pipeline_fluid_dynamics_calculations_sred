"""
tests/integration/test_main_pipeline.py
Unified End-to-End Integration Test for Navier-Stokes Execution Engine.
Executes unmocked CLI main entrypoint and validates ingestion configuration,
solver execution, state integrity, and archivist manifest/binary outputs.
"""

import json
import io
import sys
import zipfile
from pathlib import Path

import numpy as np


def test_main_full_pipeline_end_to_end(workspace_folder, monkeypatch):
    """
    Executes main() end-to-end without mocks through ingestion, C++ engine, and archivist,
    validating input/config parity, manifest structure, and binary snapshot tensor shapes.
    """
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    output_manifest_name = "navier_stokes_output.json"

    # 1. Configure CLI environment arguments
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_manifest_name,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # 2. Execute full unmocked pipeline (Ingestion -> State -> C++ Gate -> Archivist)
    from src.main import main
    main()

    # 3. Verify Output Manifest File Existence & Schema Structure
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Output JSON manifest missing at: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert "inputs" in manifest_data, "Manifest missing 'inputs' block"
    assert "config" in manifest_data, "Manifest missing 'config' block"
    assert "results" in manifest_data, "Manifest missing 'results' block"

    # 4. Verify Ingestion & Config Parity
    input_data = manifest_data["inputs"]
    config_data = manifest_data["config"]

    # Physical Constraints & Domain
    assert input_data["physical_constraints"]["min_velocity"] == -10.0
    assert input_data["physical_constraints"]["max_velocity"] == 10.0
    assert input_data["physical_constraints"]["min_pressure"] == -100.0
    assert input_data["physical_constraints"]["max_pressure"] == 100.0
    assert input_data["domain_configuration"]["type"] == "INTERNAL"
    assert input_data["domain_configuration"]["reference_velocity"] == [0.0, 0.0, 0.0]

    # Grid & Fluid Properties
    assert input_data["grid"]["nx"] == 4
    assert input_data["grid"]["ny"] == 4
    assert input_data["grid"]["nz"] == 4
    assert input_data["fluid_properties"]["density"] == 1.0
    assert input_data["fluid_properties"]["viscosity"] == 0.01

    # Simulation Parameters & Forces
    assert input_data["simulation_parameters"]["time_step"] == 0.001
    assert input_data["simulation_parameters"]["total_time"] == 0.003
    assert input_data["simulation_parameters"]["output_interval"] == 1
    assert len(input_data["boundary_conditions"]) == 1
    assert input_data["boundary_conditions"][0]["type"] == "no-slip"
    assert len(input_data["mask"]) == 64
    assert input_data["external_forces"]["gravity_vector"] == [0.0, -9.81, 0.0]

    # Production Config Integration
    assert config_data["max_poisson_iterations"] == 2000
    assert config_data["poisson_tolerance"] == 1e-8

    # 5. Verify Results Status and Timestamped Output ZIP
    results = manifest_data["results"]
    assert results["status"] == "SUCCESS", f"Expected SUCCESS status, got: {results.get('status')}"

    zip_filename = results.get("zip_filename")
    assert zip_filename and zip_filename != "NOT_APPLICABLE", f"Invalid zip_filename: {zip_filename}"

    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"Output ZIP archive missing at: {zip_path}"

    # 6. Verify C++ Generated Field Binary Snapshots (.npy) in ZIP Archive
    expected_snapshots = ["field_u.npy", "field_v.npy", "field_w.npy", "field_p.npy"]
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for snapshot in expected_snapshots:
            assert snapshot in namelist, f"Missing snapshot binary '{snapshot}' in archive. Found: {namelist}"

            # Load array directly from archive bytes and check spatial dimensions (nx=4, ny=4, nz=4)
            array_bytes = zf.read(snapshot)
            array_data = np.load(io.BytesIO(array_bytes))
            assert array_data.shape == (4, 4, 4), f"Unexpected shape {array_data.shape} for {snapshot}"
            assert not np.isnan(array_data).any(), f"NaN values detected in snapshot {snapshot}"
            assert not np.isinf(array_data).any(), f"Inf values detected in snapshot {snapshot}"
