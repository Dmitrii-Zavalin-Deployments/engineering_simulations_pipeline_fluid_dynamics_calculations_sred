"""
@file test_archivist.py
@brief Literate Test Suite for the Archivist Module

This test file serves as a narrative document and verification suite for src/archivist.py.
Explanatory text and physical/structural formulas are written as commented prose, while
executable Python assertions verify manifest serialization, C++ bridge field synchronization,
zero-padded step snapshot exports, zip archiving, error validation, and fallback state handling.
"""

import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.archivist import archive_simulation_results, export_step_snapshot


class MockSolverState:
    """Mock sovereign state container matching attributes expected by the archivist module."""

    def __init__(self, with_fields=True):
        self.input_data = {"domain": "3d_cube", "resolution": [8, 8, 8]}
        self.config = {"max_poisson_iterations": 50, "poisson_tolerance": 1e-6}
        self.current_iteration = 10
        if with_fields:
            self.fields = np.zeros((4, 8, 8, 8), dtype=np.float64)
            self.fields[0, :, :, :] = 1.0  # u component
            self.fields[1, :, :, :] = 2.0  # v component
            self.fields[2, :, :, :] = 3.0  # w component
            self.fields[3, :, :, :] = 4.0  # p component
        else:
            self.fields = None


# ============================================================================
# NARRATIVE SECTION 1: Simulation Failure Archiving
# ============================================================================
# When a simulation run terminates with a failure status, the archivist must
# bypass artifact packaging (omitting ZIP bundle creation) while still
# producing a fully schema-compliant audit manifest.
#
# The structural validation rules enforced under failure conditions are:
#     manifest["results"]["status"] == "FAILURE"
#     manifest["results"]["zip_filename"] == "NOT_APPLICABLE"
# ============================================================================


def test_archivist_failure_status_handling(workspace_folder):
    """Verifies that Archivist handles simulation failures correctly by omitting ZIP creation
    and generating a schema-compliant output manifest with zip_filename set to NOT_APPLICABLE."""
    folder = workspace_folder["folder"]
    output_filename = "failure_manifest.json"

    # We execute the archival pipeline with an explicit failure status:
    archive_simulation_results(
        state=MockSolverState(with_fields=False),
        output_dir=folder,
        output_filename=output_filename,
        status="FAILURE",
    )

    # We verify that the manifest file is correctly written to disk:
    manifest_path = Path(folder) / output_filename
    assert manifest_path.is_file()

    # We parse the serialized JSON manifest for inspection:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # The serialized metadata must reflect the failure state and disable artifact referencing:
    assert manifest["results"]["status"] == "FAILURE"
    assert manifest["results"]["zip_filename"] == "NOT_APPLICABLE"


# ============================================================================
# NARRATIVE SECTION 2: Input Parameter Validation
# ============================================================================
# The archivist strictly enforces non-null argument contracts for all parameters:
#     state != None
#     output_dir != None
#     output_filename != None
#     status != None
# Passing None to any of these parameters must trigger a fatal ValueError.
# ============================================================================


def test_archivist_null_argument_validations(workspace_folder):
    """Verifies that passing None to any required argument raises a ValueError."""
    folder = workspace_folder["folder"]
    state = MockSolverState()

    # We test explicit None checks for each required parameter:
    with pytest.raises(ValueError, match="state must be explicitly provided"):
        archive_simulation_results(
            state=None, output_dir=folder, output_filename="out.json", status="SUCCESS"
        )

    with pytest.raises(ValueError, match="output_dir must be explicitly provided"):
        archive_simulation_results(
            state=state, output_dir=None, output_filename="out.json", status="SUCCESS"
        )

    with pytest.raises(ValueError, match="output_filename must be explicitly provided"):
        archive_simulation_results(
            state=state, output_dir=folder, output_filename=None, status="SUCCESS"
        )

    with pytest.raises(ValueError, match="status must be explicitly provided"):
        archive_simulation_results(
            state=state, output_dir=folder, output_filename="out.json", status=None
        )


def test_export_step_snapshot_null_arguments():
    """Verifies that passing None to export_step_snapshot raises ValueError."""
    with pytest.raises(ValueError, match="state must be explicitly provided"):
        export_step_snapshot(None, 1, "/some/dir")

    with pytest.raises(ValueError, match="output_dir must be explicitly provided"):
        export_step_snapshot(MockSolverState(), 1, None)


# ============================================================================
# NARRATIVE SECTION 3: C++ Solver Binding Field Synchronization
# ============================================================================
# Prior to exporting field snapshots on SUCCESS, the archivist queries the
# state for an attached C++ solver instance (_cpp_solver).
#
# Case 1: Callable sync_fields(state) is invoked:
#     _cpp_solver.sync_fields(state)
# Case 2: Callable get_fields() returns updated array:
#     state.fields = np.array(_cpp_solver.get_fields(), copy=True)
# Case 3: Missing valid bindings triggers RuntimeError:
#     raise RuntimeError("... missing required 'sync_fields' or 'get_fields' binding.")
# ============================================================================


def test_archivist_cpp_solver_sync_fields(workspace_folder):
    """Verifies C++ solver field synchronization via sync_fields binding."""
    folder = workspace_folder["folder"]
    state = MockSolverState()

    class MockCppSolverSync:
        def __init__(self):
            self.synced = False

        def sync_fields(self, solver_state):
            self.synced = True
            solver_state.fields[0, 0, 0, 0] = 99.0

    cpp_solver = MockCppSolverSync()
    state._cpp_solver = cpp_solver

    archive_simulation_results(
        state=state,
        output_dir=folder,
        output_filename="cpp_sync_manifest.json",
        status="SUCCESS",
    )

    # We verify that sync_fields was executed during archival:
    assert cpp_solver.synced is True
    assert state.fields[0, 0, 0, 0] == 99.0


def test_archivist_cpp_solver_get_fields(workspace_folder):
    """Verifies C++ solver field synchronization via get_fields binding."""
    folder = workspace_folder["folder"]
    state = MockSolverState()

    updated_tensor = np.ones((4, 8, 8, 8), dtype=np.float64) * 42.0

    class MockCppSolverGet:
        def get_fields(self):
            return updated_tensor

    state._cpp_solver = MockCppSolverGet()

    archive_simulation_results(
        state=state,
        output_dir=folder,
        output_filename="cpp_get_manifest.json",
        status="SUCCESS",
    )

    # We verify state.fields was updated using get_fields output:
    assert np.allclose(state.fields, 42.0)


def test_archivist_cpp_solver_invalid_binding_raises_error(workspace_folder):
    """Verifies RuntimeError when attached C++ solver lacks sync_fields and get_fields."""
    folder = workspace_folder["folder"]
    state = MockSolverState()

    class MockCppSolverInvalid:
        pass

    state._cpp_solver = MockCppSolverInvalid()

    with pytest.raises(RuntimeError, match="missing required 'sync_fields' or 'get_fields' binding"):
        archive_simulation_results(
            state=state,
            output_dir=folder,
            output_filename="invalid_cpp_manifest.json",
            status="SUCCESS",
        )


# ============================================================================
# NARRATIVE SECTION 4: Field Resolution and Fallback Assembly
# ============================================================================
# When state.fields is None, the archivist attempts to reconstruct the 4D field
# tensor by stacking individual scalar/vector fields (u, v, w, p):
#     fields = stack([u, v, w, p], axis=0)
#
# If individual field attributes are also missing or incomplete:
#     fields == None ==> raise ValueError
# ============================================================================


def test_archivist_individual_fields_fallback(workspace_folder):
    """Verifies fallback assembly of fields array from u, v, w, p attributes and archiving into ZIP."""
    folder = workspace_folder["folder"]
    output_filename = "fallback_manifest.json"
    state = MockSolverState(with_fields=False)

    # We attach individual field components:
    grid_shape = (8, 8, 8)
    state.u = np.full(grid_shape, 1.0)
    state.v = np.full(grid_shape, 2.0)
    state.w = np.full(grid_shape, 3.0)
    state.p = np.full(grid_shape, 4.0)

    archive_simulation_results(
        state=state,
        output_dir=folder,
        output_filename=output_filename,
        status="SUCCESS",
    )

    manifest_path = Path(folder) / output_filename
    assert manifest_path.is_file()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_zip = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / target_zip
    assert zip_path.is_file()

    # Verify zero-padded step snapshot binaries were successfully archived and cleaned up from root
    with zipfile.ZipFile(zip_path, "r") as zf:
        contents = zf.namelist()
        assert "field_u_step_000010.npy" in contents
        assert "field_v_step_000010.npy" in contents
        assert "field_w_step_000010.npy" in contents
        assert "field_p_step_000010.npy" in contents

    # Verify loose files do not clutter the directory
    assert not (Path(folder) / "field_u_step_000010.npy").exists()


def test_archivist_missing_fields_raises_error(workspace_folder):
    """Verifies ValueError when neither state.fields nor individual fields are present."""
    folder = workspace_folder["folder"]
    state = MockSolverState(with_fields=False)

    with pytest.raises(ValueError, match="state.fields must be explicitly provided and populated"):
        archive_simulation_results(
            state=state,
            output_dir=folder,
            output_filename="error_manifest.json",
            status="SUCCESS",
        )


# ============================================================================
# NARRATIVE SECTION 5: End-to-End Success Archiving and ZIP Packaging
# ============================================================================
# On SUCCESS execution:
# 1. Zero-padded step snapshot files (field_u_step_*.npy, etc.) are exported or gathered.
# 2. Field snapshots are packaged into a timestamped ZIP archive:
#     archive_name = YYYYMMDD_HHMMSS.zip
# 3. Output JSON manifest is written containing status and zip filename reference.
# ============================================================================


def test_archivist_success_full_archiving_pipeline(workspace_folder):
    """Verifies full successful archiving lifecycle including step NPY creation, ZIP packaging, and JSON manifest."""
    folder = workspace_folder["folder"]
    output_filename = "success_manifest.json"
    state = MockSolverState(with_fields=True)

    archive_simulation_results(
        state=state,
        output_dir=folder,
        output_filename=output_filename,
        status="SUCCESS",
    )

    manifest_path = Path(folder) / output_filename
    assert manifest_path.is_file()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["results"]["status"] == "SUCCESS"
    target_zip = manifest["results"]["zip_filename"]
    assert target_zip.endswith(".zip")

    # Verify ZIP archive content
    zip_path = Path(folder) / target_zip
    assert zip_path.is_file()

    with zipfile.ZipFile(zip_path, "r") as zf:
        contents = zf.namelist()
        assert "field_u_step_000010.npy" in contents
        assert "field_v_step_000010.npy" in contents
        assert "field_w_step_000010.npy" in contents
        assert "field_p_step_000010.npy" in contents


# ============================================================================
# NARRATIVE SECTION 6: Loose Snapshot Binary Cleanup Verification
# ============================================================================
# When archive_simulation_results completes a SUCCESS archival run, individual
# zero-padded .npy snapshot files are compressed into a timestamped ZIP archive. 
# To prevent workspace clutter, the loose .npy files must be unlinked/deleted from disk, 
# leaving only the ZIP archive, input files, and the output JSON manifest.
# ============================================================================


def test_archivist_cleanup_loose_npy_files(workspace_folder):
    """Verifies that uncompressed temporary zero-padded .npy snapshot files are cleaned up from disk
    after being compressed into the ZIP archive."""
    folder = workspace_folder["folder"]
    output_filename = "cleanup_manifest.json"
    state = MockSolverState(with_fields=True)

    archive_simulation_results(
        state=state,
        output_dir=folder,
        output_filename=output_filename,
        status="SUCCESS",
    )

    # Verify that the JSON manifest and the ZIP archive exist
    manifest_path = Path(folder) / output_filename
    assert manifest_path.is_file()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    target_zip = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / target_zip
    assert zip_path.is_file()

    # Verify that loose uncompressed step .npy files have been removed from the output directory
    for name in ["field_u_step_000010.npy", "field_v_step_000010.npy", "field_w_step_000010.npy", "field_p_step_000010.npy"]:
        npy_file_path = Path(folder) / name
        assert not npy_file_path.exists(), f"Loose temporary file found on disk: {name}"


# ============================================================================
# NARRATIVE SECTION 7: Cleanup Exception Handling & Resilience
# ============================================================================
# During temporary file unlinking, underlying filesystem errors (such as 
# permission restrictions or file locks) may prevent deletion of .npy binaries.
# The archivist must intercept OSError exceptions gracefully, issue a diagnostic
# warning, and proceed to complete manifest writing without raising a fatal error.
# ============================================================================


def test_archivist_cleanup_oserror_handling(workspace_folder):
    """Verifies that an OSError during file unlinking is caught gracefully and logged as a warning."""
    folder = workspace_folder["folder"]
    state = MockSolverState(with_fields=True)

    # Force Path.unlink to raise an OSError when cleaning up .npy files
    with patch.object(Path, "unlink", side_effect=OSError("Simulated permission denied error")):
        archive_simulation_results(
            state=state,
            output_dir=folder,
            output_filename="oserror_manifest.json",
            status="SUCCESS",
        )

    # Manifest and ZIP packaging should still complete successfully
    manifest_path = Path(folder) / "oserror_manifest.json"
    assert manifest_path.is_file()
