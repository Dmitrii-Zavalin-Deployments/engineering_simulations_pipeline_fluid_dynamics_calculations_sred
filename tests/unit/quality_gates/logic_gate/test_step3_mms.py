# tests/quality_gates/logic_gate/test_step3_mms.py

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def _force_interior(block):
    """
    Override Step-2 dummy ghost flags so the block behaves like a true interior block.
    The dummy builder marks many blocks as boundary-adjacent (ghost neighbors),
    but MMS tests require interior behavior.
    """
    block.i_minus.is_ghost = False
    block.i_plus.is_ghost = False
    block.j_minus.is_ghost = False
    block.j_plus.is_ghost = False
    block.k_minus.is_ghost = False
    block.k_plus.is_ghost = False
    return block


def test_logic_gate_3_physics_boundary_sync():
    """
    Logic Gate 3: Physics & Boundary Sync Verification
    Ensures 'Center-Write, Neighbor-Read' architectural integrity.
    """
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    solver_input = create_validated_input(nx=nx, ny=ny, nz=nz)
    context = SimulationContext(input_data=solver_input, config=None)

    # Select block 0 but force it to behave as interior
    block = _force_interior(state.stencil_matrix[0])
    ghost_neighbor = block.i_minus

    # Poison neighbor
    ghost_neighbor.set_field(FI.VX_STAR, 1.0)

    orchestrate_step3(
        block=block,
        context=context,
        state_grid=state.grid,
        state_bc_manager=state.boundary_conditions,
        is_first_pass=True,
    )

    final_neighbor_vx = ghost_neighbor.get_field(FI.VX_STAR)
    assert final_neighbor_vx == 1.0, (
        "ARCHITECTURAL BREACH: Neighbor mutation detected! "
        f"Expected 1.0, got {final_neighbor_vx}."
    )


def test_logic_gate_3_center_mutation_audit():
    """
    Ensure the Boundary Applier mutates the cell it OWNS.
    A masked Core Block center must be sanitized by the Applier.
    """
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    context = SimulationContext(
        input_data=create_validated_input(nx=nx, ny=ny, nz=nz),
        config=None,
    )

    # Find a block with mask <= 0 OR force one
    try:
        target_block = next(b for b in state.stencil_matrix if b.center.mask <= 0)
    except StopIteration:
        target_block = state.stencil_matrix[0]
        target_block.center.mask = -1

    # Force interior behavior (disable ghost-based domain detection)
    target_block = _force_interior(target_block)

    # Poison center
    target_block.center.set_field(FI.VX_STAR, 1.0)

    orchestrate_step3(
        block=target_block,
        context=context,
        state_grid=state.grid,
        state_bc_manager=state.boundary_conditions,
        is_first_pass=True,
    )

    final_val = target_block.center.get_field(FI.VX_STAR)

    assert final_val != 1.0, (
        "MMS FAILURE: Boundary Applier ignored its own center cell. "
        f"Block {target_block.id} (Mask={target_block.center.mask}) remained poisoned."
    )
    assert final_val == 0.0, f"Expected 0.0 (No-Slip), got {final_val}"
