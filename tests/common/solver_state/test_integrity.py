# tests/common/solver_state/test_integrity.py

import logging

import numpy as np
import pytest

from src.common.field_schema import FI
from src.common.solver_state import verify_foundation_integrity

# Import your dummy helper
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


class TestSolverStateFoundation:

    @pytest.fixture
    def populated_state(self):
        """
        Rule 4 Compliance: Uses the centralized Step 1 Dummy 
        to ensure the test state matches the Archivist's expectation.
        """
        # Create a 2x2x2 hydrated state
        state = make_step1_output_dummy(nx=2, ny=2, nz=2)
        return state

    # --- 1. MEMORY INTEGRITY (POST) ---

    def test_verify_foundation_integrity_success(self, populated_state, caplog):
        """Verifies that a healthy state passes the pre-flight check."""
        caplog.set_level(logging.INFO)
        
        # Identity priming requires a stencil matrix to check against
        class MockBlock:
            def __init__(self, idx):
                self.center = type('obj', (object,), {
                    'is_ghost': False, 
                    'index': idx, 
                    'p': float(idx) + (float(FI.P) / 10.0),
                    'u': [float(idx) + (float(FI.VX) / 10.0), 0, 0]
                })
        
        # Calculate ghosted size (nx+2) * (ny+2) * (nz+2) -> 4*4*4 = 64
        num_cells = populated_state.fields.data.shape[0]
        populated_state.stencil_matrix = [MockBlock(i) for i in range(num_cells)]
        
        verify_foundation_integrity(populated_state)
        assert "Memory integrity verified" in caplog.text

    def test_verify_foundation_integrity_failure_drift(self, populated_state, caplog):
        """Verifies that the POST catches memory drift/misalignment."""
        caplog.set_level(logging.CRITICAL)
        
        class CorruptBlock:
            def __init__(self):
                self.center = type('obj', (object,), {
                    'is_ghost': False, 'index': 0, 'p': -999.9, 'u': [0, 0, 0]
                })
        
        populated_state.stencil_matrix = [CorruptBlock()]
        
        with pytest.raises(RuntimeError, match="Memory Swap"):
            verify_foundation_integrity(populated_state)

    # --- 2. ANTI-FRANKENSTEIN PROTOCOL (RULE 9) ---

    def test_rollback_recovery(self, populated_state, caplog):
        """Ensures rollback wipes 'numerical pollution'."""
        caplog.set_level(logging.WARNING)
        populated_state.capture_stable_state()
        populated_state.fields.data[:] = 144.93
        populated_state.rollback_to_stable_state()
        # Should be back to 0.0 (as initialized by dummy)
        assert np.all(populated_state.fields.data == 0.0)

    # --- 3. VECTORIZED PHYSICAL AUDIT (RULE 7) ---

    def test_audit_velocity_corridor_violation(self, populated_state):
        """Triggers the 'STABILITY TRIGGER' when velocity exceeds limits."""
        # Use limits set in dummy (-100 to 100)
        populated_state.fields.data[0, FI.VX] = 500.0 
        
        with pytest.raises(ArithmeticError, match="Velocity Corridor Violation"):
            populated_state.audit_physical_bounds()

    def test_audit_pressure_reconstruction_and_bounds(self, populated_state):
        """Verifies Real-Pressure reconstruction triggers on out-of-bounds results."""
        # Rule 7 Check: We need a localized spike so that even after 
        # subtracting the boundary mean, the peak value exceeds the corridor.
        
        # 1. Zero out the entire field (Reference boundary mean becomes 0.0)
        populated_state.fields.data[:, FI.P_NEXT] = 0.0
        
        # 2. Inject a 20M spike at a single index (Dummy limit is 1e6)
        populated_state.fields.data[0, FI.P_NEXT] = 20_000_000.0
        
        with pytest.raises(ArithmeticError, match="Pressure Corridor Violation"):
            populated_state.audit_physical_bounds()

    # --- 4. STATE TRANSITIONS ---

    def test_ready_for_time_loop_trigger(self, populated_state):
        """Verifies that ready_for_time_loop triggers checks."""
        # Will fail if stencil_matrix is missing
        populated_state.stencil_matrix = None 
        with pytest.raises(RuntimeError):
            populated_state.ready_for_time_loop = True

    def test_verify_foundation_integrity_ignores_ghosts(self, populated_state, caplog):
        """
        Validates Lines 55-57: Ensure POST skips Ghost Cells.
        We deliberately corrupt a Ghost Cell; if the loop doesn't 'continue', 
        it will trigger a Memory Drift RuntimeError.
        """
        caplog.set_level(logging.INFO)
        num_cells = populated_state.fields.data.shape[0]

        class MockBlock:
            def __init__(self, idx, is_ghost=False):
                # We create a 'center' object that mimics the solver's block structure
                self.center = type('obj', (object,), {
                    'is_ghost': is_ghost, 
                    'index': idx, 
                    # If it's a ghost, we give it 'poisoned' data that doesn't 
                    # match the Identity Priming formula.
                    'p': -999.9 if is_ghost else float(idx) + (float(FI.P) / 10.0),
                    'u': [-999.9, 0, 0] if is_ghost else [float(idx) + (float(FI.VX) / 10.0), 0, 0]
                })

        # Create a matrix where the first cell is a GHOST and the rest are REAL
        stencil = []
        stencil.append(MockBlock(0, is_ghost=True)) # The "Trap"
        for i in range(1, num_cells):
            stencil.append(MockBlock(i, is_ghost=False))
            
        populated_state.stencil_matrix = stencil

        # EXECUTION: If the 'continue' logic is broken, this raises RuntimeError
        verify_foundation_integrity(populated_state)
        
        # VERIFICATION
        assert "Memory integrity verified" in caplog.text
        # Ensure we didn't accidentally pass because there were no blocks
        assert len(populated_state.stencil_matrix) == num_cells
    
    def test_verify_foundation_integrity_failure_vx_drift(self, populated_state, caplog):
        """
        Validates Lines 66-69: Detects mismatch in Velocity (VX) pointers.
        If the block's internal 'u[0]' differs from the Primed Identity, 
        it triggers a Critical Vector Component Displacement error.
        """
        caplog.set_level(logging.CRITICAL)
        num_cells = populated_state.fields.data.shape[0]

        class DriftingVelocityBlock:
            def __init__(self, idx, should_drift=False):
                self.center = type('obj', (object,), {
                    'is_ghost': False, 
                    'index': idx, 
                    # Pressure is correct
                    'p': float(idx) + (float(FI.P) / 10.0),
                    # Velocity X is deliberately sabotaged if should_drift is True
                    'u': [
                        -888.8 if should_drift else float(idx) + (float(FI.VX) / 10.0), 
                        0.0, 
                        0.0
                    ]
                })

        # Create a matrix where the last cell has drifted
        stencil = []
        for i in range(num_cells - 1):
            stencil.append(DriftingVelocityBlock(i, should_drift=False))
        
        # Inject the drift at the final index
        stencil.append(DriftingVelocityBlock(num_cells - 1, should_drift=True))
            
        populated_state.stencil_matrix = stencil

        # EXECUTION & VERIFICATION
        expected_error = f"CRITICAL: Vector Component Displacement at Index {num_cells - 1}!"
        with pytest.raises(RuntimeError, match=expected_error):
            verify_foundation_integrity(populated_state)
            
        assert f"MEMORY DRIFT [VX]: Index {num_cells - 1}" in caplog.text