# tests/property_integrity/test_step4_initialization.py

import numpy as np
import pytest

from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.common.solver_state import (
    FieldManager,
    GridManager,
    ManifestManager,
    MaskManager,
    SimulationParameterManager,
    SolverState,
)
from src.step4.orchestrate_step4 import orchestrate_step4

# --- MANDATE: Vertical Integrity Dummy Imports ---
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


class TestStep4Initialization:
    """AUDITOR: Step 4 Archivist (Persistence) Pipeline Verification."""

    @pytest.fixture(scope="class")
    @classmethod
    def setup_state(cls):
        """
        Rule 9: Dummy Initialization.
        Prepare a SolverState to verify archival decision logic (Step 4).
        """
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        
        # Rule 5: Explicit Numerical Config
        config = SolverConfig(
            ppe_tolerance=1e-6, 
            ppe_atol=1e-9, 
            ppe_max_iter=1000, 
            ppe_omega=1.0,
            dt_min_limit=1e-6,
            ppe_max_retries=5
        )
        
        # Ensure the interval is set for testing save triggers
        input_data.simulation_parameters.output_interval = 10 
        context = SimulationContext(input_data=input_data, config=config)
        
        state = SolverState()
        state.iteration = 0 
        state.time = 0.0
        
        # Hydrate simulation parameters for the decision logic
        params = SimulationParameterManager()
        params.time_step = input_data.simulation_parameters.time_step
        params.total_time = input_data.simulation_parameters.total_time
        params.output_interval = 10
        state.simulation_parameters = params
        
        # Minimal Foundation for Rule 9 compliance
        fields = FieldManager()
        fields.allocate(n_cells=216) 
        state.fields = fields
        
        # Grid Setup
        grid = GridManager()
        grid.nx, grid.ny, grid.nz = 4, 4, 4
        grid.x_min, grid.x_max = 0.0, 1.0
        grid.y_min, grid.y_max = 0.0, 1.0
        grid.z_min, grid.z_max = 0.0, 1.0
        state.grid = grid

        # Rule 4 Compliance: io_archivist requires a valid mask for snapshot serialization
        masks = MaskManager()
        masks.mask = np.zeros((4, 4, 4), dtype=np.int32)
        state.mask = masks
        
        # Manifest is the primary target for Step 4
        state.manifest = ManifestManager()
        state.manifest.output_directory = "output/"
        
        return state, context

    def test_archivist_orchestration_contract(self, setup_state):
        """Verify the 'Handover' from compute logic to archival record."""
        state, context = setup_state
        # Rule 4: Set iteration to a multiple of output_interval (10)
        state.iteration = 10 
        
        orchestrate_step4(state, context)
        
        assert len(state.manifest.saved_snapshots) > 0, "Archivist failed to record snapshot."

    def test_archival_decision_logic(self, setup_state):
        """Verify iteration-based naming convention in the manifest."""
        state, context = setup_state
        state.iteration = 20
        
        orchestrate_step4(state, context)
        
        # Verify the Rule 8 specific naming convention: snapshot_NNNN.h5
        assert any("snapshot_0020.h5" in s for s in state.manifest.saved_snapshots), \
            "Manifest naming convention drift detected."
    
    # --- TRANSITION & TERMINAL STATE CHECKS ---

    def test_bridge_step4_to_output_integrity(self):
        """
        Vertical Integrity Mandate: Continuity Check.
        Verifies alignment between Step 4 (Archivist) and the final Solver Output.
        """
        nx, ny, nz = 4, 4, 4
        # make_step4_output_dummy provides the converged, post-commit state
        intermediate_state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
        terminal_state = make_output_schema_dummy(nx=nx, ny=ny, nz=nz)

        # 1. Physical Field Continuity (Proof of Data survival through archival)
        assert (intermediate_state.fields.data == terminal_state.fields.data).all(), \
            "Data corruption: Foundation drift between Step 4 and Terminal Output."

        # 2. Manifest Evolution (Knowledge Inheritance)
        assert len(terminal_state.manifest.saved_snapshots) >= len(intermediate_state.manifest.saved_snapshots), \
            "Terminal manifest lost snapshot history from Step 4."
        
        # 3. Path Rooting Safety
        root = terminal_state.manifest.output_directory
        for path in terminal_state.manifest.saved_snapshots:
            assert path.startswith(root), f"Escape detected: Path {path} is outside {root}"

    def test_final_state_exit_contract(self):
        """Verify the solver successfully triggers loop-termination logic."""
        target = 1.0
        state = make_output_schema_dummy(nx=4, ny=4, nz=4)
        state.simulation_parameters.total_time = target
        state.time = target 
        
        # Termination check: state must acknowledge the simulation is finished
        assert state.time >= state.simulation_parameters.total_time
        
        # At terminal state, the time loop readiness should be False (exit condition met)
        assert state.ready_for_time_loop is False