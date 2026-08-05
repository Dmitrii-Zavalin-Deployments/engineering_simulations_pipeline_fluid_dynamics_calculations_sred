from pathlib import Path
import pytest
from src.step4.generate_previews import generate_pipeline_previews


def test_generate_pipeline_previews(tmp_path: Path) -> None:
  """Test that generate_pipeline_previews successfully creates the target directory

  and exports all three diagnostic PNG files with non-zero byte size.
  """
  # Arrange: Set up a temporary output directory using pytest's tmp_path fixture
  test_output_dir = tmp_path / "testing-input-output"
  assert not test_output_dir.exists()

  # Act: Run the preview generator pointing to the temporary directory
  generate_pipeline_previews(output_dir=str(test_output_dir))

  # Assert: Verify directory was created
  assert test_output_dir.exists()
  assert test_output_dir.is_dir()

  # Assert: Verify all three required diagnostic snapshot files exist and are valid
  expected_files = [
      "initial_field_setup.png",
      "ppe_solver_state.png",
      "velocity_vorticity_slice.png",
  ]

  for filename in expected_files:
    file_path = test_output_dir / filename
    assert file_path.exists(), f"Expected preview image '{filename}' was not created."
    assert file_path.is_file(), f"Path '{filename}' is not a valid file."
    assert file_path.stat().st_size > 0, f"Generated preview image '{filename}' is empty (0 bytes)."
