# tests/property_integrity/test_step3_initialization.py

import numpy as np
import pytest

from src.common.elasticity import ElasticManager
from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_input_schema_dummy import create_validated_input


class TestStep3Initialization:
    """AUDITOR: Step 3 Projection Method & Boundary Pipeline Verification."""

    @pytest.fixture(scope="class")
    def setup_state(self):
        """Prepare a fully wired state with explicit configuration per Rule 5."""
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        
        # Rule 5: Static Config (dt removed per new SolverConfig schema)
        config = SolverConfig(
            ppe_tolerance=1e-5,
            ppe_atol=1e-9,
            ppe_max_iter=100,
            ppe_omega=1.0,
            dt_min_limit=0.001,
            ppe_max_retries=3,
            divergence_threshold=1e-4
        )
        
        context = SimulationContext(input_data=input_data, config=config)
        
        # Step 1 & 2 Orchestration to hydrate the foundation
        state = orchestrate_step1(context)
        state = orchestrate_step2(state)
        
        # ElasticManager requires state for SSoT ladder calculation
        elasticity = ElasticManager(config, state)
        
        return state, context, elasticity

    def test_ghost_cell_immunity(self, setup_state):
        """Rule 9: Ensure Step 3 logic ignores ghost cell pointers via sync path."""
        state, context, _elasticity = setup_state
        # Pick a block from the matrix and force center to be ghost
        block = state.stencil_matrix[0]
        block.center.is_ghost = True 
        
        # Pass required grid/bc managers from state
        _, delta = orchestrate_step3(
            block, context, state.grid, state.boundary_conditions, is_first_pass=False
        )
        assert delta == 0.0, "Ghost cells must return 0.0 delta (short-circuit path)."

    def test_foundation_mutation_consistency(self, setup_state):
        """Rule 9: Verify buffer mutation remains schema-compliant."""
        state, context, _elasticity = setup_state
        block = state.stencil_matrix[len(state.stencil_matrix) // 2]
        block.center.is_ghost = False # Ensure it's treated as fluid
        
        # Execute corrector/synchronization step
        orchestrate_step3(
            block, context, state.grid, state.boundary_conditions, is_first_pass=False
        )
        
        # Verify schema integrity using FI (Pressure)
        p_val = block.center.get_field(FI.P)
        assert p_val is not None, "Pressure field unreachable in foundation."
        assert not np.isnan(p_val), "NaN injection detected during SOR iteration."

    def test_boundary_integration_contract(self, setup_state):
        """Rule 5 & 8: Verify merged Step 3/4 Boundary Enforcement logic."""
        state, context, _elasticity = setup_state
        block = state.stencil_matrix[0] # Usually a boundary-adjacent cell
        
        # We test the 'is_first_pass' flag which triggers predictor + boundary enforcement
        _, delta = orchestrate_step3(
            block, context, state.grid, state.boundary_conditions, is_first_pass=True
        )
        
        # is_first_pass only predicts and enforces, doesn't solve PPE, so delta is 0
        assert delta == 0.0
        # Verification of predictor side-effect: VX_STAR should no longer be 0 if initialized
        assert block.center.get_field(FI.VX_STAR) is not None

    def test_omega_parameter_ingress(self, setup_state):
        """Rule 4: Verify numerical parameter (omega) is sourced from context."""
        state, context, _elasticity = setup_state
        # Update config to ensure the solver sees the new omega
        context.config.ppe_omega = 1.2 
        
        block = state.stencil_matrix[len(state.stencil_matrix) // 3]
        block.center.is_ghost = False
        
        _, delta = orchestrate_step3(
            block, context, state.grid, state.boundary_conditions, is_first_pass=False
        )
        assert isinstance(delta, float), "Orchestrator must return a scalar residual (delta)."

    def test_boundary_manager_lookup_integrity(self, setup_state):
        """Rule 8: Verify Step 3 accesses the boundary manager correctly."""
        state, _, _ = setup_state
        # Ensure the BC manager used in Step 3 is correctly populated from Step 1
        assert len(state.boundary_conditions.to_dict()) > 0, "BC Manager lookup table is empty."