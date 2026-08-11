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
    output_zip = workspace_folder["output_zip_name"]

    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_zip,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    from src.main import main
    main()

    zip_path = Path(folder) / output_zip
    assert zip_path.is_file()

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "output_summary.json" in namelist
        # Verify 2D cross-section preview files are packed inside ZIP
        assert any(name.startswith("previews/preview_step_") for name in namelist)

        summary_json = json.loads(zf.read("output_summary.json").decode("utf-8"))
        assert summary_json["status"] == "COMPLETED"
        assert summary_json["total_iterations_completed"] == 3


def test_full_pipeline_with_positional_cli_arg(workspace_folder, monkeypatch):
    input_path = workspace_folder["input_path"]

    cli_args = [
        "main.py",
        input_path,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    from src.main import main
    main()

    # When using positional input, default archive zip is written to workspace folder
    default_zip = Path(input_path).parent / "simulation_results.zip"
    assert default_zip.is_file()
