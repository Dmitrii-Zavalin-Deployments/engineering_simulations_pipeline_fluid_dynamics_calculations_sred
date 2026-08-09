# tests/scientific/test_pressure_configuration.py

import pytest

from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import get_explicit_solver_config

# Rule 5: Explicit Mock Configuration for testing
MOCK_CONFIG = {
    "ppe_tolerance": 1e-6,
    "ppe_atol": 1e-10,
    "ppe_max_iter": 100,
    "ppe_omega": 1.5,
    "dt_min_limit": 1e-6,
    "ppe_max_retries": 3,
    "divergence_threshold": 1e-4
}

def validate_pressure_reference_logic(state):
    """
    Logic Gate: Verifies that exactly one pressure reference exists.
    This mimics the logic required by Rule 7 for numerical stability.
    """
    # Access via the confirmed 'conditions' property in SolverState
    p_refs = sum(1 for bc in state.boundary_conditions.conditions if 'p' in bc.values)
    
    if p_refs == 0:
        raise ValueError("No pressure reference found")
    if p_refs > 1:
        raise ValueError("Over-determined pressure")

class TestPressureBoundaryLogic:
    """
    Quality Gate: Validates the 'Reference Pressure' requirement.
    Ensures the simulation is neither floating nor over-constrained.
    """

    def _create_context_with_bcs(self, bc_list):
        """Helper to inject specific BC combinations into a context."""
        input_dict = get_explicit_solver_config(nx=4, ny=4, nz=4)
        input_dict["boundary_conditions"] = bc_list
        return SimulationContext.create(input_dict, MOCK_CONFIG.copy())

    def test_pass_pressure_at_outflow_only(self):
        """Standard Case: P fixed at Outflow, V fixed at Inflow."""
        bcs = [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}},
            {"location": "x_max", "type": "outflow", "values": {"p": 0.0}}
        ]
        context = self._create_context_with_bcs(bcs)
        state = orchestrate_step1(context)
        
        # Should not raise any error
        validate_pressure_reference_logic(state)
        
        # Verify specific assignment
        outflow = next(bc for bc in state.boundary_conditions.conditions if bc.location == "x_max")
        assert "p" in outflow.values

    def test_pass_pressure_at_inflow_only(self):
        """Reverse Case: P fixed at Inflow, Outflow is floating (velocity-based)."""
        bcs = [
            {"location": "x_min", "type": "inflow", "values": {"p": 101325.0}},
            {"location": "x_max", "type": "outflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}}
        ]
        context = self._create_context_with_bcs(bcs)
        state = orchestrate_step1(context)
        
        validate_pressure_reference_logic(state)
        
        inflow = next(bc for bc in state.boundary_conditions.conditions if bc.location == "x_min")
        assert "p" in inflow.values

    def test_fail_dual_pressure_reference(self):
        """Failure Case: Over-determined (P at both ends)."""
        bcs = [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "p": 100.0}},
            {"location": "x_max", "type": "outflow", "values": {"p": 0.0}}
        ]
        context = self._create_context_with_bcs(bcs)
        state = orchestrate_step1(context)
        
        with pytest.raises(ValueError, match="Over-determined pressure"):
            validate_pressure_reference_logic(state)

    def test_fail_no_pressure_reference(self):
        """Failure Case: Under-determined (No P reference)."""
        bcs = [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}},
            {"location": "x_max", "type": "outflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}}
        ]
        context = self._create_context_with_bcs(bcs)
        state = orchestrate_step1(context)
        
        with pytest.raises(ValueError, match="No pressure reference found"):
            validate_pressure_reference_logic(state)