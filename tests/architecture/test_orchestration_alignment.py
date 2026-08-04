# tests/architecture/test_orchestration_alignment.py

import os

import pytest

# Core Logic
from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4

# Factory Functions
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy

# Rule 5: Explicit numerical settings
MOCK_CONFIG = {
    "ppe_tolerance": 1e-6,
    "ppe_atol": 1e-10,
    "ppe_max_iter": 1000,
    "ppe_omega": 1.5,
    "dt_min_limit": 1e-6,
    "ppe_max_retries": 5,
    "divergence_threshold": 1e-4,
    "dt": 0.01 
}

def assert_structural_parity(actual, expected, path=""):
    """Recursively verifies that keys and data types match (Rule 5 Audit)."""
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())
    
    assert actual_keys == expected_keys, (
        f"Structure Break at '{path}': Missing or extra keys.\n"
        f"Diff: {actual_keys.symmetric_difference(expected_keys)}"
    )
    
    for key in actual:
        actual_val = actual[key]
        expected_val = expected[key]
        current_path = f"{path}.{key}" if path else key
        
        if actual_val is None or expected_val is None:
            assert actual_val == expected_val, f"Null mismatch at {current_path}"
            continue

        assert type(actual_val) is type(expected_val), (
            f"Type Mismatch at '{current_path}': "
            f"Expected {type(expected_val).__name__}, got {type(actual_val).__name__}"
        )
        
        if isinstance(actual_val, dict):
            assert_structural_parity(actual_val, expected_val, current_path)
        elif isinstance(actual_val, list) and actual_val and isinstance(actual_val[0], dict):
            # Only recurse if the elements are dictionaries
            assert_structural_parity(actual_val[0], expected_val[0], f"{current_path}[0]")

class TestVerticalIntegrity:
    """Verifies data survival and structural alignment across the pipeline."""

    @pytest.fixture
    def global_context(self):
        NX, NY, NZ = 4, 4, 4
        # FIX: Ensure NZ and NY use their own values
        input_dummy = create_validated_input(nx=NX, ny=NY, nz=NZ)
        return SimulationContext.create(input_dummy.to_dict(), MOCK_CONFIG.copy())

    def test_input_to_step1_pipeline(self, global_context):
        """Phase 1: Validates Input -> Step 1 Orchestration"""
        expected_dummy = make_step1_output_dummy(nx=4, ny=4, nz=4)
        actual_state = orchestrate_step1(global_context)
        
        assert_structural_parity(actual_state.to_dict(), expected_dummy.to_dict())

    def test_step1_to_step2_pipeline(self, global_context):
        """Phase 2: Validates Step 1 -> Step 2 Orchestration"""
        step1_state = orchestrate_step1(global_context)
        expected_dummy = make_step2_output_dummy(nx=4, ny=4, nz=4)
        
        actual_state = orchestrate_step2(step1_state)
        
        assert_structural_parity(actual_state.to_dict(), expected_dummy.to_dict())
    
    def test_step2_to_step3_pipeline(self, global_context):
        """Phase 3: Validates Step 2 -> Step 3 Block Integrity"""
        state = orchestrate_step2(orchestrate_step1(global_context))
        
        # Target a single block
        sample_block = state.stencil_matrix[0]
        expected_block_dummy = make_step3_output_dummy() 
        
        # Execute Step 3 Physics Kernel
        actual_block, _delta = orchestrate_step3(
            block=sample_block,
            context=global_context,
            state_grid=state.grid,
            state_bc_manager=state.boundary_conditions,
            is_first_pass=True
        )
        
        assert_structural_parity(actual_block.to_dict(), expected_block_dummy.to_dict())

    def test_step3_to_step4_pipeline(self, global_context):
        """Phase 4: Validates Step 3 -> Step 4 (Archivist Persistence)"""
        # Step 4 is the final stage. It manages serialization to disk.
        state = orchestrate_step2(orchestrate_step1(global_context))
        
        # Rule 4: Set state to trigger an archival event
        state.iteration = 10 
        global_context.input_data.simulation_parameters.output_interval = 10
        
        if not os.path.exists("output"): 
            os.makedirs("output")

        actual_state = orchestrate_step4(state, global_context)
        
        # Verify the Archivist updated the manifest correctly
        assert len(actual_state.manifest.saved_snapshots) > 0, "Archivist failed to update manifest"
        assert "snapshot_0010.h5" in actual_state.manifest.saved_snapshots[-1]

    def test_full_pipeline_continuity(self, global_context):
        """Verifies the entire chain from Step 1 through Step 4."""
        # 1. Initialize State
        state = orchestrate_step1(global_context)
        
        # 2. Assemble Stencils
        state = orchestrate_step2(state)
        
        # 3. Process Physics (Simulate one iteration for the first block)
        # We ensure the data doesn't lose structure after a predictor pass
        _block, _ = orchestrate_step3(
            block=state.stencil_matrix[0],
            context=global_context,
            state_grid=state.grid,
            state_bc_manager=state.boundary_conditions,
            is_first_pass=True
        )
        
        # 4. Final Archival check
        state.iteration = 0
        final_state = orchestrate_step4(state, global_context)
        
        assert final_state.ready_for_time_loop is True
        assert final_state.grid.nx == 4
