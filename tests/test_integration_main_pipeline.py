"""
tests/test_integration_main_pipeline.py
Unified End-to-End Integration Test for Navier-Stokes Execution Engine.
Executes unmocked CLI main entrypoint and validates ingestion configuration,
solver execution, state integrity, field drift/parity, archivist output artifacts,
and Pybind11 C++/Python memory bridge pointer preservation on a 3x3x3 grid.
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
    input_path = Path(folder) / input_file
    output_manifest_name = "navier_stokes_output.json"

    # 1. Update input JSON to apply non-zero external forces, initial velocity,
    #    and schema-compliant boundary conditions using the nested 'values' dictionary.
    with open(input_path, "r", encoding="utf-8") as f:
        input_json_data = json.load(f)

    input_json_data["external_forces"]["force_vector"] = [1.0, 0.0, 0.0]
    input_json_data["boundary_conditions"] = [{"location": "x_min", "type": "pressure", "values": {"p": 10.0}}]
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

    # 3. Execute full unmocked pipeline (Ingestion -> State -> C++ Gate -> Archivist)
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

    # Physical Constraints & Domain
    assert input_data["physical_constraints"]["min_velocity"] == -10.0
    assert input_data["physical_constraints"]["max_velocity"] == 10.0
    assert input_data["physical_constraints"]["min_pressure"] == -100.0
    assert input_data["physical_constraints"]["max_pressure"] == 100.0
    assert input_data["domain_configuration"]["type"] == "INTERNAL"
    assert input_data["domain_configuration"]["reference_velocity"] == [0.0, 0.0, 0.0]

    # Grid & Fluid Properties (Compact 3x3x3)
    assert input_data["grid"]["nx"] == 3
    assert input_data["grid"]["ny"] == 3
    assert input_data["grid"]["nz"] == 3
    assert input_data["fluid_properties"]["density"] == 1.0
    assert input_data["fluid_properties"]["viscosity"] == 0.01

    # Simulation Parameters & Forces
    assert input_data["simulation_parameters"]["time_step"] == 0.001
    assert input_data["simulation_parameters"]["total_time"] == 0.003
    assert input_data["simulation_parameters"]["output_interval"] == 1
    assert len(input_data["boundary_conditions"]) == 1
    assert input_data["boundary_conditions"][0]["type"] == "pressure"
    assert input_data["boundary_conditions"][0]["values"]["p"] == 10.0
    assert len(input_data["mask"]) == 27  # 3 x 3 x 3 = 27 cells
    assert input_data["external_forces"]["force_vector"] == [1.0, 0.0, 0.0]
    assert input_data["external_forces"]["gravity_vector"] == [0.0, -9.81, 0.0]

    # Production Config Integration
    assert config_data["max_poisson_iterations"] == 2000
    assert config_data["poisson_tolerance"] == 1e-8

    # 6. Verify Results Status and Timestamped Output ZIP
    results = manifest_data["results"]
    assert results["status"] == "SUCCESS", f"Expected SUCCESS status, got: {results.get('status')}"

    zip_filename = results.get("zip_filename")
    assert zip_filename and zip_filename != "NOT_APPLICABLE", f"Invalid zip_filename: {zip_filename}"

    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"Output ZIP archive missing at: {zip_path}"

    # 7. Verify C++ Generated Field Binary Snapshots (.npy) in ZIP Archive
    final_step = 3
    expected_snapshots = [
        f"field_u_step_{final_step:06d}.npy",
        f"field_v_step_{final_step:06d}.npy",
        f"field_w_step_{final_step:06d}.npy",
        f"field_p_step_{final_step:06d}.npy",
    ]
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for snapshot in expected_snapshots:
            assert snapshot in namelist, f"Missing snapshot binary '{snapshot}' in archive. Found: {namelist}"

            # Load array directly from archive bytes and check spatial dimensions (3, 3, 3)
            array_bytes = zf.read(snapshot)
            array_data = np.load(io.BytesIO(array_bytes))
            assert array_data.shape == (3, 3, 3), f"Unexpected shape {array_data.shape} for {snapshot}"
            assert not np.isnan(array_data).any(), f"NaN values detected in snapshot {snapshot}"
            assert not np.isinf(array_data).any(), f"Inf values detected in snapshot {snapshot}"

            # DYNAMIC FIELD EVOLUTION VERIFICATION:
            # Under dynamic driving forces and initial velocity, field_u, field_v, field_w, and field_p
            # MUST exhibit non-zero values to verify solver execution and C++ memory mutation.
            assert np.max(np.abs(array_data)) > 0.0, (
                f"CRITICAL ERROR: {snapshot} is identically zero. "
                "C++ solver failed to mutate field or transfer memory."
            )


def test_python_cpp_field_state_parity(workspace_folder):
    """
    Verifies zero-drift parity between Python SolverState in-memory numpy fields
    and C++ exported binary snapshots written to the archived ZIP container.
    """
    from src.archivist import archive_simulation_results
    from src.cpp_gate import step_simulation
    from src.ingestion import load_and_validate_inputs
    from src.state import SolverState

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "parity_test_output.json"

    # Drive non-zero dynamic evolution
    input_data, config_data = load_and_validate_inputs(input_path, Path(folder) / "config.json")
    input_data["external_forces"]["force_vector"] = [1.0, 1.0, 1.0]
    input_data["initial_conditions"]["velocity"] = [0.1, 0.1, 0.1]

    state = SolverState(input_data, config_data)
    step_simulation(state)

    # Export via Archivist
    archive_simulation_results(state, folder, output_manifest_name, status="SUCCESS")

    manifest_path = Path(folder) / output_manifest_name
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    zip_filename = manifest_data["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename

    # Validate memory state exact equality with serialized binary snapshots
    field_names = ["field_u", "field_v", "field_w", "field_p"]
    final_step = state.current_iteration
    with zipfile.ZipFile(zip_path, "r") as zf:
        for idx, name in enumerate(field_names):
            snapshot_filename = f"{name}_step_{final_step:06d}.npy"
            array_bytes = zf.read(snapshot_filename)
            archived_array = np.load(io.BytesIO(array_bytes))
            memory_array = state.fields[idx]

            # Enforce exact floating point equality across memory and disk snapshot
            np.testing.assert_array_equal(
                memory_array,
                archived_array,
                err_msg=f"Memory/Disk drift detected for field {name}!",
            )


def test_pybind11_memory_bridge_forensic_audit(workspace_folder):
    """
    Forensic audit test verifying Pybind11 C++/Python memory bridge integrity.
    Confirms in-place buffer mutation without pointer reallocation and asserts
    non-zero mutations across u, v, w, and p fields under dynamic body forces.
    """
    from src.cpp_gate import step_simulation
    from src.ingestion import load_and_validate_inputs
    from src.state import SolverState

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file

    input_data, config_data = load_and_validate_inputs(input_path, Path(folder) / "config.json")
    input_data["external_forces"]["force_vector"] = [1.0, 2.0, 1.5]
    input_data["initial_conditions"]["velocity"] = [0.2, -0.1, 0.3]
    input_data["boundary_conditions"] = [{"location": "x_min", "type": "pressure", "values": {"p": 5.0}}]

    state = SolverState(input_data, config_data)

    # 1. Capture initial memory addresses of numpy array buffers
    initial_pointers = [field.ctypes.data for field in state.fields]

    # 2. Execute C++ solver step
    step_simulation(state)

    # 3. Verify in-place memory pointer preservation (no reallocation during C++ execution)
    post_pointers = [field.ctypes.data for field in state.fields]
    field_labels = ["field_u", "field_v", "field_w", "field_p"]

    for name, pre_ptr, post_ptr in zip(field_labels, initial_pointers, post_pointers):
        assert pre_ptr == post_ptr, (
            f"MEMORY DRIFT DETECTED: Pointer for {name} shifted from {hex(pre_ptr)} "
            f"to {hex(post_ptr)}. C++ solver reallocated memory instead of in-place mutation."
        )

    # 4. Verify all four fields (u, v, w, p) received non-zero C++ mutations
    for idx, name in enumerate(field_labels):
        field_data = state.fields[idx]
        assert np.max(np.abs(field_data)) > 0.0, (
            f"FIELD MUTATION ERROR: {name} is identically zero after solver step. "
            "Pybind11 bridge failed to write mutated values back to Python state."
        )
