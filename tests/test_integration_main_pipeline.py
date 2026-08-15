"""
tests/test_integration_main_pipeline.py
Unified End-to-End Integration Test for Navier-Stokes Execution Engine.
Executes unmocked CLI main entrypoint and validates ingestion configuration,
solver execution, state integrity, field drift/parity, and archivist output artifacts.
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


def test_main_full_pipeline_end_to_end(workspace_folder, monkeypatch):
    """
    Executes main() end-to-end without mocks through ingestion, C++ engine, and archivist,
    validating input/config parity, manifest structure, physical field evolution, and binary shapes.
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

            # PHYSICAL EVOLUTION VERIFICATION:
            # Prevent false-positive passes on all-zero uninitialized buffers.
            # Gravity is g_y = -9.81 m/s^2, so field_v MUST exhibit negative vertical acceleration.
            if snapshot == "field_v.npy":
                assert np.max(np.abs(array_data)) > 0.0, (
                    "CRITICAL ERROR: field_v is identically zero. C++ solver failed to mutate field or transfer memory."
                )
                assert np.min(array_data) < 0.0, (
                    f"CRITICAL ERROR: field_v minimum value ({np.min(array_data)}) is non-negative. "
                    "Field failed to respond to gravity g_y = -9.81."
                )


def test_python_cpp_field_state_parity(workspace_folder):
    """
    Verifies zero-drift parity between Python SolverState in-memory numpy fields
    and C++ exported binary snapshots written to the archived ZIP container.
    """
    from src.archivist import archive_simulation_results
    from src.cpp_gate import execute_cpp_solver
    from src.ingestion import ingest_simulation_data
    from src.state import SolverState

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    output_manifest_name = "parity_test_output.json"

    # Ingest data & run solver directly to hold SolverState handle
    input_data, config_data = ingest_simulation_data(folder, input_file, "config.json")
    state = SolverState(input_data, config_data)
    execute_cpp_solver(state)

    # Export via Archivist
    archive_simulation_results(state, folder, output_manifest_name, status="SUCCESS")

    manifest_path = Path(folder) / output_manifest_name
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    zip_filename = manifest_data["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename

    # Validate memory state exact equality with serialized binary snapshots
    field_names = ["field_u", "field_v", "field_w", "field_p"]
    with zipfile.ZipFile(zip_path, "r") as zf:
        for idx, name in enumerate(field_names):
            snapshot_filename = f"{name}.npy"
            array_bytes = zf.read(snapshot_filename)
            archived_array = np.load(io.BytesIO(array_bytes))
            memory_array = state.fields[idx]

            # Enforce exact floating point equality across memory and disk snapshot
            np.testing.assert_array_equal(
                memory_array,
                archived_array,
                err_msg=f"Memory/Disk drift detected for field {name}!",
            )
