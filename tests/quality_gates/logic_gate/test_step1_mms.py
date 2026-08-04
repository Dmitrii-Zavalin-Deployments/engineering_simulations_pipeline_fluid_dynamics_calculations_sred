# tests/quality_gates/logic_gate/test_step1_mms.py

import numpy as np

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import create_validated_input


def wrap_in_context(solver_input):
    """
    Compliance: Rule 4. Wraps raw data in the appropriate 
    SimulationContext to avoid __slots__ AttributeErrors.
    """
    return SimulationContext(input_data=solver_input, config=None)

def test_logic_gate_1_padded_ingestion():
    """
    Logic Gate 1: Padded Ingestion Verification
    
    Analytical Challenge: Padded Masking
    Success Metric: (N+2)^3 Size Parity
    Compliance: Rule 9 (Topological Mask Unified into Foundation)
    Compliance: Rule 4 (SSoT Hierarchy - No Facade Properties)
    """

    # 1. Setup Input: Explicitly define grid to avoid Hidden Defaults (Rule 5)
    nx, ny, nz = 2, 2, 2
    solver_input = create_validated_input() 
    
    # Pathing Fix: SolverInput has direct access to .grid
    solver_input.grid.nx = nx
    solver_input.grid.ny = ny
    solver_input.grid.nz = nz
    
    # Compliance: Wrap in SimulationContext to simulate main.py loop
    context = wrap_in_context(solver_input)
    
    # Success Metric Calculation: (2+2)*(2+2)*(2+2) = 64 cells
    # This reflects the Ghost Cell Padding requirement for the Foundation.
    expected_cells = (nx + 2) * (ny + 2) * (nz + 2) 

    # 2. Action: orchestrate_step1(context)
    # This hydrates the FieldManager (Foundation) and flattens the padded 3D mask.
    state = orchestrate_step1(context)

    # 3. Verification: (N+2)^3 Size Parity (Rule 1: Field Precision/Scale Audit)
    actual_cells = state.fields.data.shape[0]
    assert actual_cells == expected_cells, (
        f"MMS FAILURE [Size Parity]: Expected {expected_cells} cells, got {actual_cells}. "
        "Check Rule 9 Ghost Padding logic."
    )
    
    # 4. Verification: Schema Consistency (Rule 1 & Rule 7)
    actual_width = state.fields.data.shape[1]
    expected_width = FI.num_fields()
    assert actual_width == expected_width, (
        f"MMS FAILURE [Schema]: Foundation width {actual_width} "
        f"does not match FI schema {expected_width}."
    )

    # 5. Verification: Padded Masking Logic
    # SSoT Check: Mask data must reside in state.fields.data, not a facade property.
    mask_buffer = state.fields.data[:, FI.MASK]
    assert mask_buffer.ndim == 1, "MMS FAILURE: Mask buffer must be a 1D slice of the Foundation."
    
    # Verification: Ghost Cell Padding (0.0)
    unique_vals = np.unique(mask_buffer)
    assert 0.0 in unique_vals, (
        "MMS FAILURE [Padded Masking]: Ghost cell padding (0.0) missing from Foundation buffer."
    )

    # 6. Verification: Logic-Layer Hydration & SSoT Compliance (Rule 4)
    assert hasattr(state, 'fields'), "SSoT BREACH: state.fields (Physical Context) missing."
    assert hasattr(state, 'grid'), "SSoT BREACH: state.grid (Geometric Context) missing."
    
    # ARCHITECTURAL CHECK: No facade properties allowed on SolverState
    assert not hasattr(state, 'nx'), "Rule 4 Violation: Found 'nx' alias on SolverState."
    assert not hasattr(state, 'density'), "Rule 4 Violation: Found 'density' alias on SolverState."