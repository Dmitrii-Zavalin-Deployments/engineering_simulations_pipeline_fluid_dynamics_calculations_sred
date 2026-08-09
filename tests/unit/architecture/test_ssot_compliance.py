# tests/architecture/test_ssot_compliance.py

from src.common.solver_state import SolverState


def test_solver_state_attribute_purity():
    """
    Rule 4 Audit: Ensure SolverState does not contain 'Facade' properties.
    Allowed attributes are only the core containers and internal dunder methods.
    """
    # Define the ONLY allowed top-level attributes for SolverState
    # If you add a new legitimate container, add it here.
    ALLOWED_CONTAINERS = {
        'config', 
        'grid', 
        'fields', 
        'iteration', 
        'manifest', 
        'ready_for_time_loop',
        'stencil_matrix', # For Step 2/3 transitions
        'boundary_conditions',
        'external_forces',
        'simulation_parameters',
        'physical_constraints',
        'fluid_properties',
        'initial_conditions',
        'time',
        'mask',
        'domain_configuration'
    }
    
    # Get all public attributes (those not starting with _)
    current_attributes = {
        attr for attr in dir(SolverState) 
        if not attr.startswith('_') and not callable(getattr(SolverState, attr))
    }
    
    # Check for pollution (properties that should be inside grid, fields, or config)
    pollution = current_attributes - ALLOWED_CONTAINERS
    
    assert not pollution, (
        f"🚨 SSoT BREACH: Unauthorized attributes found in SolverState: {pollution}. "
        "Move these into their respective sub-containers (grid, fields, or config)."
    )