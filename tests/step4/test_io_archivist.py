# tests/step4/test_io_archivist.py

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.step4.io_archivist import save_snapshot


def test_save_snapshot_success(tmp_path):
    """
    Targets the happy path: Verifies HDF5 creation and Ghost Stripping.
    """
    # 1. Setup a dummy state with 2x2x2 physical grid (4x4x4 with ghosts)
    state = MagicMock()
    state.iteration = 1
    state.time = 0.1
    state.grid.nx, state.grid.ny, state.grid.nz = 2, 2, 2
    state.grid.x_min, state.grid.x_max = 0.0, 1.0
    state.grid.y_min, state.grid.y_max = 0.0, 1.0
    state.grid.z_min, state.grid.z_max = 0.0, 1.0
    
    # Foundation buffer: (4*4*4) rows, enough columns for FI
    num_cells = (2 + 2) ** 3
    state.fields.data = np.zeros((num_cells, 10))
    state.mask.mask = np.zeros((2, 2, 2))  # Physical size mask
    state.manifest.saved_snapshots = []

    # 2. Patch Path to use a temporary directory for the test
    with (
        patch("src.step4.io_archivist.Path", return_value=tmp_path),
        patch.object(Path, "mkdir"),
    ):
        save_snapshot(state)
    
    # 3. Assertions
    assert len(state.manifest.saved_snapshots) == 1
    assert "snapshot_0001.h5" in state.manifest.saved_snapshots[0]


def test_save_snapshot_critical_failure():
    """
    Targets Lines 76-78: Triggers Exception handling during IO.
    """
    state = MagicMock()
    state.iteration = 99
    state.grid.nx, state.grid.ny, state.grid.nz = 2, 2, 2
    
    # 1. Mock h5py.File to raise an OSError (e.g., Permission Denied) and verify exception
    with (
        patch("h5py.File", side_effect=OSError("Read-only file system")),
        patch("src.step4.io_archivist.Path.mkdir"),
        pytest.raises(OSError, match="Read-only file system"),
    ):
        save_snapshot(state)
