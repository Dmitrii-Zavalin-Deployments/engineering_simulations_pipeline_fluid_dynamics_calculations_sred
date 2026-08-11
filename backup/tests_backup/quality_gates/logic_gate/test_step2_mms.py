# tests/quality_gates/logic_gate/test_step2_mms.py

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def test_logic_gate_2_stencil_graph_assembly():
    """
    Logic Gate 2: Stencil Graph Assembly Verification
    
    Analytical Challenge: Graph Integrity
    Success Metric: 7-Point Connectivity
    Target: src/step2/orchestrate_step2.py
    """
    # 1. Setup Input: Hydrate SolverState prototype using atomic constructor injection
    # We use a small 4x4x4 grid for deterministic index verification
    state_in = make_step1_output_dummy(nx=4, ny=4, nz=4)

    # 2. Action: orchestrate_step2(state)
    # This triggers assemble_stencil_matrix which encapsulates the registry
    state_out = orchestrate_step2(state_in)

    # 3. Verification: Graph Integrity (Success Metric)
    # Check that the stencil_matrix was attached to the state
    assert hasattr(state_out, 'stencil_matrix'), "MMS FAILURE: Step 2 failed to attach stencil_matrix to state"
    
    # 4. Verification: 7-Point Connectivity
    # Verify a core block has non-null neighbors in all 6 directions (i, j, k)
    # For a 4x4x4 grid with padding, core indices start after the ghost layer.
    core_blocks = state_out.stencil_matrix
    assert len(core_blocks) > 0, "MMS FAILURE: Stencil matrix is empty"
    
    sample_block = core_blocks[0]
    
    # Verify the logic-layer correctly wired 1D indices to 3D neighbors
    assert sample_block.i_plus is not None, "Connectivity Failure: i_plus is missing"
    assert sample_block.i_minus is not None, "Connectivity Failure: i_minus is missing"
    assert sample_block.j_plus is not None, "Connectivity Failure: j_plus is missing"
    assert sample_block.j_minus is not None, "Connectivity Failure: j_minus is missing"
    assert sample_block.k_plus is not None, "Connectivity Failure: k_plus is missing"
    assert sample_block.k_minus is not None, "Connectivity Failure: k_minus is missing"

    # 5. Verification: State Transition
    # Success Metric: The state must be flagged as ready for the main compute loop
    assert state_out.ready_for_time_loop is True, "MMS FAILURE: ready_for_time_loop flag not set"

    # 6. Verification: Physics Parameter Mapping
    # Verify that FluidProperties (Density/Viscosity) were passed down to StencilBlocks
    assert sample_block.rho == state_in.fluid_properties.density, "Data Integrity: Density mismatch in block"