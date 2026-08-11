# tests/property_integrity/test_step1_initialization.py

import numpy as np
import pytest

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import create_validated_input


class TestStep1Initialization:
    """AUDITOR: Step 1 Structural Gate, Metadata Hydration & Lifecycle Transitions."""

    @pytest.fixture(scope="class")
    @classmethod
    def setup_data(cls):
        # 1. Hydrate input data (Rule 5: Explicit initialization)
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        
        # 2. Rule 5: Static Config initialization. 
        # Note: 'dt' is omitted here as it's a dynamic simulation parameter.
        config = SolverConfig(
            ppe_tolerance=1e-6, 
            ppe_atol=1e-9, 
            ppe_max_iter=1000, 
            ppe_omega=1.0, 
            dt_min_limit=0.001,
            ppe_max_retries=3,
            divergence_threshold=1e6
        )
        
        context = SimulationContext(input_data=input_data, config=config)
        state = orchestrate_step1(context)
        
        return state, context

    # --- STRUCTURAL & CONTAINER CHECKS ---

    def test_departmental_containers(self, setup_data):
        """Rule 4: Validates existence of required sub-managers and metadata."""
        state, _ = setup_data
        assert state.grid is not None, "Missing GridManager"
        assert state.fields is not None, "Missing FieldManager"
        assert state.mask is not None, "Missing MaskManager"
        
        # Checking for the specific manager type defined in orchestrate_step1
        assert hasattr(state, "simulation_parameters"), "Missing SimulationParameterManager"
        assert hasattr(state, "physical_constraints"), "Missing PhysicalConstraintsManager"

    def test_no_convenience_leaks(self, setup_data):
        """Rule 4: Ensures no convenience aliases exist on root."""
        state, _ = setup_data
        # These must be accessed via state.grid.nx or state.simulation_parameters.time_step
        forbidden = ["nx", "ny", "nz", "dt", "density", "ppe_tolerance"]
        for alias in forbidden:
            assert not hasattr(state, alias), f"Rule 4 Violation: Alias '{alias}' found on state root."

    # --- MEMORY ALLOCATION & MAPPING ---

    def test_memory_allocation_geometry(self, setup_data):
        """Rule 0 & 1: Verifies FieldManager allocation and C-contiguity."""
        state, _ = setup_data
        
        assert state.fields.data is not None, "Field data foundation is None."
        assert state.fields.data.ndim == 2, "Foundation must be 2D (n_cells, fields)."
        assert state.fields.data.flags['C_CONTIGUOUS'], "Memory foundation must be C-contiguous."
        
        # Rule 1: Buffered grid size (Ghost cells included)
        n_cells = (state.grid.nx + 2) * (state.grid.ny + 2) * (state.grid.nz + 2)
        expected_shape = (n_cells, FI.num_fields())
        assert state.fields.data.shape == expected_shape, "Foundation shape mismatch with buffered grid."
    
    def test_identity_signature_integrity(self, setup_data):
        """
        Rule 9: Identity Priming. Ensures FI schema maps to the correct buffer columns.
        """
        state, _ = setup_data
        data = state.fields.data
        
        # Populate with unique identifiable floats
        for field_id in FI:
            data[:, field_id] = np.arange(data.shape[0]) + (float(field_id) / 10.0)
            
        test_idx = data.shape[0] // 2
        # Verify P and VX columns are distinct and correctly indexed
        assert np.isclose(data[test_idx, FI.P], test_idx + (FI.P / 10.0))
        assert np.isclose(data[test_idx, FI.VX], test_idx + (FI.VX / 10.0))

    # --- PHYSICAL CONSTRAINTS & HYDRATION ---

    def test_physical_constraints_hydration(self, setup_data):
        """Verifies new constraints block is correctly mapped from input."""
        state, _ = setup_data
        constraints = state.physical_constraints
        assert constraints.min_velocity == -100.0
        assert constraints.max_pressure == 1e6

    def test_initial_conditions_persistence(self, setup_data):
        """Rule 9: Ensure mask hydration survives orchestration."""
        state, _ = setup_data
        # FI.MASK was hydrated with padded_mask.flatten() in orchestrate_step1
        assert np.any(state.fields.data[:, FI.MASK] != 0.0), "Topology mask was not hydrated."

    def test_termination_math_precision(self):
        """Validates floating point exit condition: current_time >= total_time."""
        total_time, dt = 0.05, 0.01
        
        # Manually assemble context to test boundary math
        input_data = create_validated_input(nx=4)
        input_data.simulation_parameters.total_time = total_time
        input_data.simulation_parameters.time_step = dt
        
        state = orchestrate_step1(SimulationContext(
            input_data=input_data, 
            config=SolverConfig(
                ppe_tolerance=1e-6, 
                ppe_atol=1e-12, 
                ppe_max_iter=100, 
                ppe_omega=1.0,
                dt_min_limit=0.001,
                ppe_max_retries=3
            )
        ))
        
        current_time, iterations = 0.0, 0
        # Use pytest.approx for the loop condition to mirror real-world solver logic
        while current_time < (state.simulation_parameters.total_time - 1e-12):
            current_time += state.simulation_parameters.time_step
            iterations += 1
            
        assert iterations == 5
        assert current_time == pytest.approx(total_time)