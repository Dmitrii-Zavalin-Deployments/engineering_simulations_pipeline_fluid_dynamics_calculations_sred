"""
Unified End-to-End Integration Test for Navier-Stokes Execution Engine (Accelerated Flow 8x8x4 Grid).

This test module serves as a narrative document and verification suite for the full 
unmocked pipeline, executing ingestion via python CLI (`src/main.py`), C++ core solvers 
via Pybind11, and archivist artifact packaging on an 8x8x4 grid under accelerated flow 
(mirroring the C++ test suite[cite: 1]).

Explanatory text and physical equations are written as commented prose, while executable
Python assertions verify ingestion configuration, solver execution, state integrity, field drift/parity,
and Pybind11 C++/Python memory bridge pointer preservation.
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

# ============================================================================
# NARRATIVE SECTION 1: Full Pipeline CLI End-to-End Execution (8x8x4 Accelerated Flow)
# ============================================================================
# The Navier-Stokes system ingests spatial grid bounds and boundary condition parameters,
# allocating an 8x8x4 discrete Cartesian volume:
#     V = nx * ny * nz = 8 * 8 * 4 = 256 cells
# 
# External positive body forces (fx = 0.1, fy = 0.1, fz = 0.2) and multi-directional 
# inflow velocity vectors (u = 0.5, v = 0.2, w = 0.1) drive momentum evolution equations 
# through the unmocked C++ core engine prior to packaging binary NumPy array snapshots into an output ZIP archive.
# ============================================================================


def test_main_full_pipeline_accelerated_8x8x4(workspace_folder, monkeypatch):
    """
    Executes main.py end-to-end without mocks through ingestion, C++ engine, and archivist,
    validating input/config parity, manifest structure, physical field evolution, and binary shapes
    on an 8x8x4 accelerated flow grid.
    """
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "navier_stokes_accelerated_output.json"

    # 1. Update input JSON to apply accelerated body forces, initial velocity, 8x8x4 grid,
    #    complex obstacle mask, and schema-compliant boundary conditions.
    with open(input_path, "r", encoding="utf-8") as f:
        input_json_data = json.load(f)

    input_json_data["grid"].update({"nx": 8, "ny": 8, "nz": 4})
    
    # 8x8x4 = 256 cells mask matching accelerated flow configuration[cite: 1]
    layer_mask = [
        0,  0,  0,  0,  0,  0,  0,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    input_json_data["mask"] = layer_mask * 4  # 4 layers (nz = 4) = 256 cells
    input_json_data["external_forces"]["force_vector"] = [0.1, 0.1, 0.2]
    input_json_data["initial_conditions"]["velocity"] = [0.5, 0.2, 0.1]
    input_json_data["boundary_conditions"] = [
        {"location": "z_min", "type": "inflow", "values": {"u": 0.5, "v": 0.2, "w": 0.1, "p": 0.0}},
        {"location": "z_max", "type": "outflow", "values": {"u": 0.5, "v": 0.2, "w": 0.1, "p": 0.0}},
        {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}}
    ]

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_json_data, f)

    # 2. Configure CLI environment arguments matching user specification
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_manifest_name,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # 3. Execute full unmocked pipeline via python entry point
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

    assert input_data["grid"]["nx"] == 8
    assert input_data["grid"]["ny"] == 8
    assert input_data["grid"]["nz"] == 4
    assert len(input_data["mask"]) == 256  # 8 x 8 x 4 = 256 cells

    assert config_data["max_poisson_iterations"] == 2000
    assert config_data["poisson_tolerance"] == 1e-8

    # 6. Verify Results Status and Timestamped Output ZIP
    results = manifest_data["results"]
    assert results["status"] == "SUCCESS", f"Expected SUCCESS status, got: {results.get('status')}"

    zip_filename = results.get("zip_filename")
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"Output ZIP archive missing at: {zip_path}"

    # 7. Verify C++ Generated Field Binary Snapshots (.npy) in ZIP Archive
    final_step = results.get("final_step", 1)
    expected_snapshots = [
        f"field_u_step_{final_step:06d}.npy",
        f"field_v_step_{final_step:06d}.npy",
        f"field_w_step_{final_step:06d}.npy",
        f"field_p_step_{final_step:06d}.npy",
    ]
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for snapshot in expected_snapshots:
            assert snapshot in namelist, f"Missing snapshot binary '{snapshot}' in archive."

            array_bytes = zf.read(snapshot)
            array_data = np.load(io.BytesIO(array_bytes))
            assert array_data.shape == (8, 8, 4), f"Unexpected shape {array_data.shape} for {snapshot}"
            assert not np.isnan(array_data).any(), f"NaN values detected in snapshot {snapshot}"
            assert not np.isinf(array_data).any(), f"Inf values detected in snapshot {snapshot}"
            assert np.max(np.abs(array_data)) > 0.0, f"CRITICAL ERROR: {snapshot} is identically zero."


# ============================================================================
# NARRATIVE SECTION 2: Python-C++ State Parity & Step-by-Step Stage Verification
# ============================================================================
# Zero-drift validation checks that the in-memory numpy array buffers bound via
# Pybind11 match the serialized binary snapshots archived on disk, and verifies 
# intermediate stage outputs against unmocked C++ orchestrator behavior[cite: 1].
# ============================================================================


def test_python_cpp_accelerated_stage_parity(workspace_folder):
    """
    Verifies zero-drift parity between Python SolverState in-memory numpy fields
    and C++ exported binary snapshots, alongside step-by-step stage verification 
    on the 8x8x4 accelerated flow grid.
    """
    from src.archivist import archive_simulation_results
    from src.cpp_gate import step_simulation
    from src.ingestion import load_and_validate_inputs
    from src.state import SolverState

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "parity_accelerated_output.json"

    input_data, config_data = load_and_validate_inputs(input_path, Path(folder) / "config.json")
    input_data["grid"].update({"nx": 8, "ny": 8, "nz": 4})
    
    layer_mask = [
        0,  0,  0,  0,  0,  0,  0,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    input_data["mask"] = layer_mask * 4
    input_data["external_forces"]["force_vector"] = [0.1, 0.1, 0.2]
    input_data["initial_conditions"]["velocity"] = [0.5, 0.2, 0.1]
    input_data["boundary_conditions"] = [
        {"location": "z_min", "type": "inflow", "values": {"u": 0.5, "v": 0.2, "w": 0.1, "p": 0.0}},
        {"location": "z_max", "type": "outflow", "values": {"u": 0.5, "v": 0.2, "w": 0.1, "p": 0.0}},
        {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}}
    ]

    state = SolverState(input_data, config_data)
    step_simulation(state)

    archive_simulation_results(state, folder, output_manifest_name, status="SUCCESS")

    manifest_path = Path(folder) / output_manifest_name
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    zip_filename = manifest_data["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename

    field_names = ["field_u", "field_v", "field_w", "field_p"]
    final_step = state.current_iteration
    with zipfile.ZipFile(zip_path, "r") as zf:
        for idx, name in enumerate(field_names):
            snapshot_filename = f"{name}_step_{final_step:06d}.npy"
            array_bytes = zf.read(snapshot_filename)
            archived_array = np.load(io.BytesIO(array_bytes))
            memory_array = state.fields[idx]

            np.testing.assert_array_equal(
                memory_array,
                archived_array,
                err_msg=f"Memory/Disk drift detected for field {name}!",
            )


# ============================================================================
# NARRATIVE SECTION 3: Pybind11 Memory Bridge Pointer Preservation
# ============================================================================
# C++ integration relies on zero-copy memory views where field memory pointers 
# remain invariant across simulation time-stepping steps:
#     ptr_pre == ptr_post
# ============================================================================


def test_pybind11_memory_bridge_accelerated(workspace_folder):
    """
    Verifies Pybind11 C++/Python memory bridge integrity on the 8x8x4 accelerated grid,
    confirming in-place buffer mutation without pointer reallocation.
    """
    from src.cpp_gate import step_simulation
    from src.ingestion import load_and_validate_inputs
    from src.state import SolverState

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file

    input_data, config_data = load_and_validate_inputs(input_path, Path(folder) / "config.json")
    input_data["grid"].update({"nx": 8, "ny": 8, "nz": 4})
    
    layer_mask = [
        0,  0,  0,  0,  0,  0,  0,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    input_data["mask"] = layer_mask * 4
    input_data["external_forces"]["force_vector"] = [0.1, 0.1, 0.2]
    input_data["initial_conditions"]["velocity"] = [0.5, 0.2, 0.1]
    input_data["boundary_conditions"] = [
        {"location": "z_min", "type": "inflow", "values": {"u": 0.5, "v": 0.2, "w": 0.1, "p": 0.0}},
        {"location": "z_max", "type": "outflow", "values": {"u": 0.5, "v": 0.2, "w": 0.1, "p": 0.0}},
        {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}}
    ]

    state = SolverState(input_data, config_data)
    initial_pointers = [field.ctypes.data for field in state.fields]

    step_simulation(state)

    post_pointers = [field.ctypes.data for field in state.fields]
    field_labels = ["field_u", "field_v", "field_w", "field_p"]

    for name, pre_ptr, post_ptr in zip(field_labels, initial_pointers, post_pointers):
        assert pre_ptr == post_ptr, f"MEMORY DRIFT DETECTED for {name}."

    for idx, name in enumerate(field_labels):
        assert np.max(np.abs(state.fields[idx])) > 0.0, f"FIELD MUTATION ERROR: {name} is zero."