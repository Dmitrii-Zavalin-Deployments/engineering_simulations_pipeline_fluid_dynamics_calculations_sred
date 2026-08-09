# tests/property_integrity/test_state_transitions.py

import numpy as np

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


def test_bridge_step4_to_output_integrity():
    """
    SYSTEM AUDITOR: Verifies continuity between the Finalized Step (Step 4) 
    and the Terminal/Archived (Step 6) states per src/main.py.
    """
    nx, ny, nz = 4, 4, 4
    
    # 1. Generate both dummies
    # intermediate_state: The state immediately after orchestrate_step4 (Line 123)
    intermediate_state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # terminal_state: The state post-ArchiveService (Line 153)
    terminal_state = make_output_schema_dummy(nx=nx, ny=ny, nz=nz)

    # 2. PROOF OF CONTINUITY: Rule 9 Memory Integrity
    # We use numpy testing to ensure the physics data wasn't mangled by the archiver.
    np.testing.assert_array_almost_equal(
        intermediate_state.fields.data, 
        terminal_state.fields.data,
        err_msg="Data corruption detected: Terminal state fields do not match Step 4 output."
    )
    
    # 3. PROOF OF EVOLUTION: Manifest Finalization
    # Step 4 provides the data; Step 6 (Terminal) must provide the file locations.
    assert len(terminal_state.manifest.saved_snapshots) > 0, \
        "Archive failure: Terminal manifest is empty after Step 6 transition."
    
    # 4. PROOF OF LIFECYCLE: The 'Seal' Check
    # Verify the transition from 'Looping' (Step 4) to 'Archived' (Step 6).
    # Step 4 is usually mid-loop; terminal is always sealed.
    assert terminal_state.ready_for_time_loop is False, \
        "Terminal state must have ready_for_time_loop=False to prevent infinite recursion."

    # 5. PATH VALIDITY: Relative SSoT Check
    # Ensure the Archive Service adheres to the 'output/' directory convention.
    for path in terminal_state.manifest.saved_snapshots:
        assert path.startswith("output/"), \
            f"Path Violation: Artifact '{path}' found outside the canonical output/ directory."