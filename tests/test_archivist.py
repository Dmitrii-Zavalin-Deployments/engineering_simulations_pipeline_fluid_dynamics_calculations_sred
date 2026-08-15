"""
tests/test_archivist.py
Unit tests for the Archivist module, verifying manifest serialization,
error handling, and failure status archiving behaviors.
"""

import json
from pathlib import Path


def test_archivist_failure_status_handling(workspace_folder):
    """
    Verifies that Archivist handles simulation failures correctly by omitting ZIP creation
    and generating schema-compliant output manifest with zip_filename set to NOT_APPLICABLE.
    """
    from src.archivist import archive_simulation_results

    class MockFailureState:
        def __init__(self):
            self.input_data = {"test": "data"}
            self.config = {"param": 1}
            self.fields = []

    folder = workspace_folder["folder"]
    output_filename = "failure_manifest.json"

    archive_simulation_results(
        state=MockFailureState(),
        output_dir=folder,
        output_filename=output_filename,
        status="FAILURE",
    )

    manifest_path = Path(folder) / output_filename
    assert manifest_path.is_file()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["results"]["status"] == "FAILURE"
    assert manifest["results"]["zip_filename"] == "NOT_APPLICABLE"
