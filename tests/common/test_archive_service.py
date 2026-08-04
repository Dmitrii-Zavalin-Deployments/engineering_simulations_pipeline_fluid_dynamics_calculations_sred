# tests/common/test_archive_service.py

import logging
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.main import BASE_DIR

from src.common.archive_service import archive_simulation_artifacts


class TestArchiveServiceIntegrity:

    def test_archival_and_forensic_logging(self, tmp_path, caplog):
        """
        Validates the Instance-Optimized Archiver:
        1. Correctly packages simulation data.
        2. Places final ZIP in the SSoT 'data/testing-input-output' folder.
        3. Emits explicit Forensic Logs (Rule 5).
        4. Leaves staging folder intact for instance dissolution (Rule 8).
        """
        caplog.set_level(logging.INFO)

        # --- 1. SETUP MOCK STATE ---
        mock_output_dir = tmp_path / "simulation_output_raw"
        mock_output_dir.mkdir()
        
        (mock_output_dir / "step_0.csv").write_text("u,v,w,p\n0,0,0,1")
        (mock_output_dir / "mesh.vtk").write_text("DATASET STRUCTURED_GRID")

        class MockManifest:
            def __init__(self, out_dir):
                self.output_directory = str(out_dir)

        class MockState:
            def __init__(self, out_dir):
                self.manifest = MockManifest(out_dir)

        state = MockState(mock_output_dir)

        # Define expected paths
        target_dir = Path(BASE_DIR) / "data" / "testing-input-output"
        staging_dir = Path.cwd() / "navier_stokes_output"
        expected_zip_path = target_dir / "navier_stokes_output.zip"

        # --- 2. EXECUTION ---
        result_path_str = archive_simulation_artifacts(state)
        result_path = Path(result_path_str)

        # --- 3. VERIFICATION: ARCHIVE INTEGRITY ---
        assert result_path.exists(), "Final ZIP was not created."
        assert result_path.resolve() == expected_zip_path.resolve(), "Target path mismatch."
        
        with zipfile.ZipFile(result_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            assert any("step_0.csv" in f for f in file_list), "CSV missing from ZIP."
            assert any("mesh.vtk" in f for f in file_list), "VTK missing from ZIP."

        # --- 4. VERIFICATION: SOURCE CLEANUP ---
        # The raw output should be gone because it was moved to staging
        assert not mock_output_dir.exists(), "Raw source directory was not moved."

        # --- 5. VERIFICATION: FORENSIC LOGS ---
        assert "Initiating artifact packaging" in caplog.text
        assert "Source moved to staging" in caplog.text
        assert "ARCHIVE COMPLETE" in caplog.text

        # --- 6. VERIFICATION: INSTANCE PERSISTENCE (Rule 8) ---
        # We assert that the staging folder remains, confirming the ephemeral-only design.
        assert staging_dir.exists(), "Archiver logic error: Staging folder should persist."

    def test_archive_simulation_artifacts_source_missing_error(self):
        """
        Coverage for lines 36-37: Ensure CRITICAL log and FileNotFoundError 
        when the source_dir does not exist. Added 'self' argument.
        """
        # 1. Setup Mock SolverState
        mock_state = MagicMock()
        mock_state.manifest.output_directory = "/tmp/non_existent_solver_results_9999"
        
        # 2. Setup Mock for main_solver.BASE_DIR
        with patch("src.main.BASE_DIR", "/tmp/mock_base"):
            
            # 3. Execution & Verification
            with pytest.raises(FileNotFoundError) as exc_info:
                archive_simulation_artifacts(mock_state)
            
            assert "Source directory" in str(exc_info.value)
            assert "/tmp/non_existent_solver_results_9999" in str(exc_info.value)

    def test_archive_simulation_artifacts_full_path_success(self, tmp_path, mocker):
        """
        Ensures lines 58-63 are covered. Added 'self' argument.
        """
        # Create a real dummy source directory in pytest's temp folder
        source = tmp_path / "raw_results"
        source.mkdir()
        (source / "data.txt").write_text("simulation results")
        
        # Mock SolverState
        mock_state = MagicMock()
        mock_state.manifest.output_directory = str(source)
        
        # Mock main_solver.BASE_DIR to point to our temp test area
        mocker.patch("src.main.BASE_DIR", str(tmp_path))
        
        # Execute
        final_zip = archive_simulation_artifacts(mock_state)
        
        # Verify
        assert Path(final_zip).exists()
        assert final_zip.endswith(".zip")
        assert "navier_stokes_output.zip" in final_zip
    
    def test_archive_simulation_artifacts_overwrite_logic(self, tmp_path, mocker, caplog):
        """
        Coverage for lines 60-61: Verify that an existing archive is detected,
        logged as a debug message, and unlinked before the new move.
        """
        caplog.set_level(logging.DEBUG) # Must be DEBUG to see line 60
        
        # 1. SETUP: Create a fake project structure
        mock_base = tmp_path / "project_root"
        mock_base.mkdir()
        mocker.patch("src.main.BASE_DIR", str(mock_base))
        
        # 2. SETUP: Create the "Target" and a pre-existing ZIP
        target_dir = mock_base / "data" / "testing-input-output"
        target_dir.mkdir(parents=True)
        old_archive = target_dir / "navier_stokes_output.zip"
        old_archive.write_text("old content") # This file must exist to trigger line 60
        
        # 3. SETUP: Create the "Source"
        source_dir = tmp_path / "new_results"
        source_dir.mkdir()
        (source_dir / "new_data.txt").write_text("new results")
        
        mock_state = MagicMock()
        mock_state.manifest.output_directory = str(source_dir)
        
        # 4. EXECUTION
        final_zip_str = archive_simulation_artifacts(mock_state)
        
        # 5. VERIFICATION
        assert Path(final_zip_str).exists()
        # Verify line 60 log message
        assert "Overwriting existing archive" in caplog.text
        # Verify the file was replaced (content check)
        import zipfile
        with zipfile.ZipFile(final_zip_str, 'r') as z:
            assert "new_data.txt" in z.namelist()