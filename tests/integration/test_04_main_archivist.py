"""
tests/integration/test_04_main_archivist.py
Integration Test 4: Main CLI -> Ingestion -> State -> C++ Gate -> Archivist packaging.
"""

import json
import sys
import zipfile
from pathlib import Path


def test_main_cli_archivist_stage(workspace_folder, monkeypatch):
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    output_manifest_name = workspace_folder.get(
        "output_file_name", 
        workspace_folder.get("output_zip_name", "navier_stokes_output.json")
    )

    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_manifest_name,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    from src.main import main
    main()

    # Step 1: Verify the output JSON manifest exists on disk
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.exists(), f"Output JSON manifest missing at: {manifest_path}"

    # Step 2: Validate JSON manifest structure against schema contract
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert "inputs" in manifest_data, "Manifest missing 'inputs' section"
    assert "config" in manifest_data, "Manifest missing 'config' section"
    assert "results" in manifest_data, "Manifest missing 'results' section"

    results = manifest_data["results"]
    assert results["status"] == "SUCCESS", f"Expected 'SUCCESS', got: {results.get('status')}"

    zip_filename = results.get("zip_filename")
    assert zip_filename and zip_filename != "NOT_APPLICABLE", f"Invalid zip_filename: {zip_filename}"

    # Step 3: Verify the timestamped ZIP archive exists in output directory
    expected_zip = Path(folder) / zip_filename
    assert expected_zip.exists(), f"Timestamped archive ZIP missing at: {expected_zip}"

    # Step 4: Validate ZIP archive contents (field_u.npy, field_v.npy, field_w.npy, field_p.npy)
    with zipfile.ZipFile(expected_zip, "r") as zf:
        namelist = zf.namelist()
        expected_snapshots = ["field_u.npy", "field_v.npy", "field_w.npy", "field_p.npy"]
        for snapshot_file in expected_snapshots:
            assert snapshot_file in namelist, (
                f"Missing snapshot '{snapshot_file}' in zip archive. Found: {namelist}"
            )
