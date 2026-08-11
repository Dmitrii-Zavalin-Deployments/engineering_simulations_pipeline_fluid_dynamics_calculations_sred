"""
tests/integration/test_04_main_archivist.py
Integration Test 4: Main CLI -> Ingestion -> State -> C++ Gate -> Archivist packaging.
"""

import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch


def test_main_cli_archivist_stage(workspace_folder, monkeypatch):
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

    # Bypass image preview generation to isolate archivist pipeline
    with patch("src.main.generate_snapshot_preview", return_value="mock_preview.png"):
        from src.main import main
        main()

    expected_zip = Path(folder) / output_zip
    assert expected_zip.exists()

    # Validate output archive zip contents
    with zipfile.ZipFile(expected_zip, "r") as zf:
        namelist = zf.namelist()
        assert "output_summary.json" in namelist

        summary_data = json.loads(zf.read("output_summary.json").decode("utf-8"))
        assert summary_data["status"] == "COMPLETED"
        assert summary_data["total_iterations_completed"] == 3
