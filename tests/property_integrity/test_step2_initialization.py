# tests/property_integrity/test_step2_initialization.py

import pytest

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


class TestStep2Initialization:
    """AUDITOR: Step 2 Wiring & Matrix Assembly Verification."""

    @pytest.fixture(scope="class")
    @classmethod
    def assembled_state(cls):
        """
        Hydrates state via the Step 1 Dummy and runs Step 2 orchestration.
        Verifies that Step 2 successfully ingests the Step 1 Foundation.
        """
        # 1. Fetch the immutable core baseline (The Dummy) 
        # nx=4 results in (4+2)^3 = 216 total cells, but 4^3 = 64 fluid cells.
        input_state = make_step1_output_dummy(nx=4, ny=4, nz=4)
        
        # 2. Execute Orchestration (The Implementation)
        return orchestrate_step2(input_state)

    def test_stencil_matrix_existence(self, assembled_state):
        """Rule 9: Verifies that the matrix was successfully assigned to the state."""
        assert assembled_state.stencil_matrix is not None, "Orchestrator failed to assign stencil_matrix."
        assert isinstance(assembled_state.stencil_matrix, list), "Stencil matrix must be a collection of cell stencils."
        assert len(assembled_state.stencil_matrix) > 0, "Stencil matrix was initialized empty."

    def test_readiness_sentinel_activation(self, assembled_state):
        """Rule 9: Ensures the readiness gate is engaged post-assembly."""
        # This confirms that orchestrate_step2 correctly transitioned the state
        # allowing it to pass the 'Logic Gate' into the time-stepping loop.
        assert assembled_state.ready_for_time_loop is True

    def test_matrix_dimensions(self, assembled_state):
        """Rule 0 & 5: Matrix length must match the internal fluid volume."""
        # The Stencil Matrix typically handles the calculation for the internal 
        # 'active' cells. For a 4x4x4 grid, this is 64. 
        # The FieldManager foundation handles the 216 ghosted cells.
        
        nx = assembled_state.grid.nx
        ny = assembled_state.grid.ny
        nz = assembled_state.grid.nz
        
        expected_fluid_cells = nx * ny * nz
        actual_matrix_size = len(assembled_state.stencil_matrix)
        
        assert actual_matrix_size == expected_fluid_cells, \
            f"Dimension Mismatch: Matrix({actual_matrix_size}) != Fluid Volume({expected_fluid_cells})."

    def test_physical_integrity_persistence(self, assembled_state):
        """Rule 4: Verify physical managers are not corrupted during matrix assembly."""
        # Ensure Step 2 didn't accidentally wipe managers during orchestration
        assert assembled_state.physical_constraints.min_velocity == -100.0
        assert assembled_state.fluid_properties.viscosity == 0.001