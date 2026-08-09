# tests/quality_gates/logic_gate/test_step4_mms.py

from unittest.mock import patch

from src.common.simulation_context import SimulationContext
from src.step4.orchestrate_step4 import orchestrate_step4
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_logic_gate_4_archival_trigger_logic():
    """
    Logic Gate 4: Archival Trigger Verification
    
    Analytical Challenge: Snapshot Cadence (Deterministic Triggering)
    Success Metric: I mod Delta I == 0
    Compliance: Rule 4 (SSoT - Hierarchy over Convenience)
    Compliance: Rule 5 (No Logical Defaults - Explicit iteration/time)
    """

    # 1. Setup: Explicit Input to avoid "Silent Failure" (Rule 5)
    output_interval = 10
    solver_input = create_validated_input(nx=4, ny=4, nz=4)
    
    # Pathing Fix: Ensure simulation_parameters are set on the input object
    solver_input.simulation_parameters.output_interval = output_interval

    # Compliance Rule 4: Wrap in SimulationContext (The "Main Loop Context")
    context = SimulationContext(input_data=solver_input, config=None)

    # 2. Setup: Use Step 2 dummy to get a full SolverState (The "Wiring")
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    
    # 3. Scenario 1: Positive Trigger (state.iteration = 20)
    # Rule 5: Explicit iteration setting
    state.iteration = 20
    
    # Action: Verify that orchestrate_step4 triggers the archivist (20 % 10 == 0)
    # Patch the save_snapshot utility to verify the trigger logic.
    with patch('src.step4.orchestrate_step4.save_snapshot') as mock_save:
        orchestrate_step4(state, context)
        
        # Success Metric: Triggered because 20 % 10 == 0
        mock_save.assert_called_once_with(state)
        
    # 4. Scenario 2: Negative Trigger (state.iteration = 21)
    # Test that iteration 21 does NOT trigger a save (Drift Protection).
    state.iteration = 21
    with patch('src.step4.orchestrate_step4.save_snapshot') as mock_save_neg:
        orchestrate_step4(state, context)
        
        # Success Metric: No Save because 21 % 10 != 0
        mock_save_neg.assert_not_called()

    # 5. Verification: Logic Layer Integrity (Rule 4)
    # Ensure the orchestrator returns the SolverState for the next time-loop iteration.
    result = orchestrate_step4(state, context)
    
    assert isinstance(result, type(state)), (
        "MMS FAILURE: Orchestrator failed to return the SolverState container."
    )
    
    # Rule 4 Audit: Ensure no convenience aliases were added to the result
    assert not hasattr(result, 'output_interval'), (
        "Rule 4 Violation: Facade property 'output_interval' detected on state. "
        "Must remain in context.input_data.simulation_parameters."
    )

    # 6. Verification: Deterministic Persistence (Rule 5)
    # Verify the state returned still holds the explicit iteration value.
    assert result.iteration == 21, "MMS FAILURE: State mutation/persistence lost in Step 4."