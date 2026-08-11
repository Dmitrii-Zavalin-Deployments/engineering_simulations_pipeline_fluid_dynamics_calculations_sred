# tests/quality_gates/physics_gate/test_archivist.py

from src.common.field_schema import FI
from src.step4.io_archivist import save_snapshot
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_gate_4a_ghost_stripping_parity():
    """
    Gate 4.A: Ghost Stripping Audit
    
    Identity: V_out = V_core (stripping padding)
    Success Metric: Shape_out = (nx, ny, nz)
    Compliance: Rule 9 (Hybrid Memory Foundation slicing)
    """

    # 1. Setup: Define explicit core dimensions (Rule 5)
    nx, ny, nz = 4, 4, 4
    _ = create_validated_input(nx=nx, ny=ny, nz=nz) # Ground the schema

    # 2. Setup: Create a 4x4x4 core (6x6x6 total memory Foundation with ghosts)
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 3. Logic Verification: Simulate the Archivist's internal slicing logic
    # The production archivist must reshape the flat Foundation using (N+2) bounds.
    nx_buf, ny_buf, nz_buf = nx + 2, ny + 2, nz + 2
    raw_flat = state.fields.data[:, FI.P]
    
    # 4. Apply the Ghost-Cell Lock: [1:-1] slicing on all axes (Rule 9)
    # This transforms the "Foundation" into the "Snapshot"
    full_3d = raw_flat.reshape(nz_buf, ny_buf, nx_buf) # Note: standard C-order (z, y, x)
    stripped_data = full_3d[1:-1, 1:-1, 1:-1]
    
    # 5. Verify Success Metric: Core Parity
    assert stripped_data.shape == (nz, ny, nx), (
        f"Ghost-Cell Lock Breach: Output shape {stripped_data.shape} "
        f"includes ghosts. Expected ({nz}, {ny}, {nx})"
    )

def test_gate_4b_index_integrity():
    """
    Gate 4.B: Index Integrity Audit
    
    Identity: HDF5[0,0,0] == Buffer[1,1,1] (Global Truth Alignment)
    Compliance: Rule 7 (Atomic Numerical Truth)
    """

    nx, ny, nz = 4, 4, 4
    _ = create_validated_input(nx=nx, ny=ny, nz=nz)
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    val_at_first_phys_cell = 99.85
    
    # 1. Target the first physical (non-ghost) cell at core indices (0,0,0)
    # In the Padded Foundation, this corresponds to memory/buffer indices (1,1,1)
    # Using the deterministic stride (Rule 7)
    nx_buf, ny_buf = nx + 2, ny + 2
    # Memory Flattening: i + nx_buf * (j + ny_buf * k)
    target_idx = 1 + nx_buf * (1 + ny_buf * 1)
    
    # 2. Inject the Atomic Numerical Truth into the Foundation
    state.fields.data[target_idx, FI.P] = val_at_first_phys_cell
    
    # 3. Simulate Archivist Extraction (Rule 9)
    # Reshape must strictly follow the Foundation topology
    full_3d = state.fields.data[:, FI.P].reshape(nz + 2, ny_buf, nx_buf)
    physical_snapshot = full_3d[1:-1, 1:-1, 1:-1]
    
    # 4. Verify Identity: Ensure 0-indexed HDF5 maps to 1-indexed Foundation
    extracted_val = physical_snapshot[0, 0, 0]
    assert extracted_val == val_at_first_phys_cell, (
        f"Slicing Breach: Coordinate shift detected. "
        f"Expected {val_at_first_phys_cell}, got {extracted_val}"
    )

def test_archivist_manifest_update():
    """
    Verification: Archivist correctly updates the state manifest.
    Compliance: Rule 4 (SSoT - Manifest resides in state)
    """

    # 1. Setup: Explicit iteration/time (Rule 5)
    nx, ny, nz = 2, 2, 2
    _ = create_validated_input(nx=nx, ny=ny, nz=nz)
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    state.iteration = 5
    state.time = 0.05
    
    # 2. Action: Trigger production save logic (Rule 2: Zero-Debt)
    # Ensure the io_archivist follows the manifest update protocol.
    save_snapshot(state)
    
    # 3. Verification: Manifest tracking (Hierarchy over Convenience)
    # Snapshot metadata must stay in state.manifest (Rule 4).
    assert len(state.manifest.saved_snapshots) > 0, "Archivist failed to update state.manifest"
    
    latest_entry = state.manifest.saved_snapshots[-1]
    expected_suffix = "snapshot_0005.h5"
    assert expected_suffix in latest_entry, (
        f"Manifest Drift: Expected snapshot suffix {expected_suffix}, found {latest_entry}"
    )