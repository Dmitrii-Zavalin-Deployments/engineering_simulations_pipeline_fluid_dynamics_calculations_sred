"""
tests/test_integration_cubic_grid.py
Unified End-to-End Integration Test for Navier-Stokes Execution Engine (Cubic Grid).
Executes unmocked CLI main entrypoint and validates ingestion configuration,
solver execution, state integrity, field drift/parity, archivist output artifacts,
and Pybind11 C++/Python memory bridge pointer preservation on a 4x4x4 cubic grid.
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


def test_main_full_pipeline_cubic_4x4x4(workspace_folder, monkeypatch):
    """
    Executes main() end-to-end without mocks through ingestion, C++ engine, and archivist,
    validating input/config parity, manifest structure, physical field evolution, and binary shapes
    on a symmetric 4x4x4 cubic grid.
    """
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "navier_stokes_cubic_output.json"

    # 1. Update input JSON to apply non-zero external forces, initial velocity, and 4x4x4 grid
    with open(input_path, "r", encoding="utf-8") as f:
        input_json_data = json.load(f)

    input_json_data["grid"] = {"nx": 4, "ny": 4, "nz": 4}
    input_json_data["mask"] = [0] * 64  # 4 x 4 x 4 = 64 cells
    input_json_data["external_forces"]["force_vector"] = [1.0, 1.0, 1.0]
    input_json_data["initial_conditions"]["velocity"] = [0.1, 0.1, 0.1]

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_json_data, f)

    # 2. Configure CLI environment arguments
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_manifest_name,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # 3. Execute full unmocked pipeline
    from src.main import main
    main()

    # 4. Verify Output Manifest File Existence & Schema Structure
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Output JSON manifest missing at: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert "inputs" in manifest_data, "Manifest missing 'inputs' block"
    assert "config" in manifest_data, "Manifest missing 'config' block"
    assert "results" in manifest_data, "Manifest missing 'results' block"

    # 5. Verify Ingestion & Config Parity
    input_data = manifest_data["inputs"]
    config_data = manifest_data["config"]

    assert input_data["grid"]["nx"] == 4
    assert input_data["grid"]["ny"] == 4
    assert input_data["grid"]["nz"] == 4
    assert len(input_data["mask"]) == 64  # 4 x 4 x 4 = 64 cells

    assert config_data["max_poisson_iterations"] == 2000
    assert config_data["poisson_tolerance"] == 1e-8

    # 6. Verify Results Status and Timestamped Output ZIP
    results = manifest_data["results"]
    assert results["status"] == "SUCCESS", f"Expected SUCCESS status, got: {results.get('status')}"

    zip_filename = results.get("zip_filename")
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"Output ZIP archive missing at: {zip_path}"

    # 7. Verify C++ Generated Field Binary Snapshots (.npy) in ZIP Archive
    expected_snapshots = ["field_u.npy", "field_v.npy", "field_w.npy", "field_p.npy"]
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for snapshot in expected_snapshots:
            assert snapshot in namelist, f"Missing snapshot binary '{snapshot}' in archive."

            array_bytes = zf.read(snapshot)
            array_data = np.load(io.BytesIO(array_bytes))
            assert array_data.shape == (4, 4, 4), f"Unexpected shape {array_data.shape} for {snapshot}"
            assert not np.isnan(array_data).any(), f"NaN values detected in snapshot {snapshot}"
            assert not np.isinf(array_data).any(), f"Inf values detected in snapshot {snapshot}"
            assert np.max(np.abs(array_data)) > 0.0, f"CRITICAL ERROR: {snapshot} is identically zero."


def test_python_cpp_field_state_parity_cubic(workspace_folder):
    """
    Verifies zero-drift parity between Python SolverState in-memory numpy fields
    and C++ exported binary snapshots on a 4x4x4 cubic grid.
    """
    from src.archivist import archive_simulation_results
    from src.cpp_gate import step_simulation
    from src.ingestion import load_and_validate_inputs
    from src.state import SolverState

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "parity_cubic_output.json"

    input_data, config_data = load_and_validate_inputs(input_path, Path(folder) / "config.json")
    input_data["grid"] = {"nx": 4, "ny": 4, "nz": 4}
    input_data["mask"] = [0] * 64
    input_data["external_forces"]["force_vector"] = [1.0, 1.0, 1.0]
    input_data["initial_conditions"]["velocity"] = [0.1, 0.1, 0.1]

    state = SolverState(input_data, config_data)
    step_simulation(state)

    archive_simulation_results(state, folder, output_manifest_name, status="SUCCESS")

    manifest_path = Path(folder) / output_manifest_name
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    zip_filename = manifest_data["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename

    field_names = ["field_u", "field_v", "field_w", "field_p"]
    with zipfile.ZipFile(zip_path, "r") as zf:
        for idx, name in enumerate(field_names):
            snapshot_filename = f"{name}.npy"
            array_bytes = zf.read(snapshot_filename)
            archived_array = np.load(io.BytesIO(array_bytes))
            memory_array = state.fields[idx]

            np.testing.assert_array_equal(
                memory_array,
                archived_array,
                err_msg=f"Memory/Disk drift detected for field {name}!",
            )


def test_pybind11_memory_bridge_cubic(workspace_folder):
    """
    Verifies Pybind11 C++/Python memory bridge integrity on a 4x4x4 cubic grid,
    confirming in-place buffer mutation without pointer reallocation.
    """
    from src.cpp_gate import step_simulation
    from src.ingestion import load_and_validate_inputs
    from src.state import SolverState

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file

    input_data, config_data = load_and_validate_inputs(input_path, Path(folder) / "config.json")
    input_data["grid"] = {"nx": 4, "ny": 4, "nz": 4}
    input_data["mask"] = [0] * 64
    input_data["external_forces"]["force_vector"] = [1.0, 2.0, 1.5]
    input_data["initial_conditions"]["velocity"] = [0.2, -0.1, 0.3]

    state = SolverState(input_data, config_data)
    initial_pointers = [field.ctypes.data for field in state.fields]

    step_simulation(state)

    post_pointers = [field.ctypes.data for field in state.fields]
    field_labels = ["field_u", "field_v", "field_w", "field_p"]

    for name, pre_ptr, post_ptr in zip(field_labels, initial_pointers, post_pointers):
        assert pre_ptr == post_ptr, f"MEMORY DRIFT DETECTED for {name}."

    for idx, name in enumerate(field_labels):
        assert np.max(np.abs(state.fields[idx])) > 0.0, f"FIELD MUTATION ERROR: {name} is zero."
