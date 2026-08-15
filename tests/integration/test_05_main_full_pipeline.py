"""
tests/integration/test_05_main_full_pipeline.py
Integration Test 5: Complete End-to-End Control Plane / Data Plane integration test.
"""

import json
import sys
import zipfile
from pathlib import Path


def test_full_pipeline_with_named_cli_args(workspace_folder, monkeypatch):
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

    # 1. Verify JSON manifest output
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Output JSON manifest missing: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert "results" in manifest_data, "Manifest missing 'results' block"
    results = manifest_data["results"]
    assert results["status"] == "SUCCESS", f"Expected SUCCESS status, got {results.get('status')}"

    zip_filename = results.get("zip_filename")
    assert zip_filename and zip_filename != "NOT_APPLICABLE", f"Invalid zip_filename: {zip_filename}"

    # 2. Verify timestamped ZIP archive and field snapshot binaries
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"Timestamped output ZIP archive missing: {zip_path}"

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        expected_snapshots = ["field_u.npy", "field_v.npy", "field_w.npy", "field_p.npy"]
        for snapshot in expected_snapshots:
            assert snapshot in namelist, f"Missing {snapshot} in archive. Found: {namelist}"


def test_full_pipeline_with_positional_cli_arg(workspace_folder, monkeypatch):
    input_path = workspace_folder["input_path"]

    cli_args = [
        "main.py",
        input_path,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    from src.main import main
    main()

    # Search parent directory for output JSON manifest or generated zip archive
    parent_dir = Path(input_path).parent
    manifest_candidates = list(parent_dir.glob("*.json"))
    assert len(manifest_candidates) > 0, f"No JSON manifest output generated in {parent_dir}"

    manifest_path = manifest_candidates[0]
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    zip_filename = manifest_data.get("results", {}).get("zip_filename")
    assert zip_filename and zip_filename != "NOT_APPLICABLE"

    default_zip = parent_dir / zip_filename
    assert default_zip.is_file(), f"Default output ZIP file missing: {default_zip}"