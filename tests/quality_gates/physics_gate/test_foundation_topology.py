# tests/quality_gates/physics_gate/test_foundation_topology.py

from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.stencil_assembler import assemble_stencil_matrix
from tests.helpers.solver_input_schema_dummy import create_validated_input


def test_gate_1a_foundation_volume_parity():
    """
    Gate 1.A: Foundation Volume Audit
    
    Identity: V_total = (nx+2)(ny+2)(nz+2)
    Compliance: Rule 9 (Hybrid Memory Foundation allocation)
    """

    # 1. Setup: Explicit dimensions via SSoT path (Rule 5)
    nx, ny, nz = 4, 4, 4
    
    # FIX: Pass dimensions directly to ensure mask data matches grid volume
    solver_input = create_validated_input(nx=nx, ny=ny, nz=nz)
    
    # Pathing Fix: Ensure grid object properties are explicitly aligned
    solver_input.grid.nx = nx
    solver_input.grid.ny = ny
    solver_input.grid.nz = nz

    # Compliance: Wrap in SimulationContext SSoT container
    context = SimulationContext(input_data=solver_input, config=None)
    
    # The Foundation must allocate space for the core + 2 ghost layers per dimension
    expected_volume = (nx + 2) * (ny + 2) * (nz + 2)
    
    # 2. Action: Orchestrate Step 1 (Hydrate FieldManager)
    state = orchestrate_step1(context)
    
    # 3. Verification: Access Physical Context Foundation (Rule 4)
    actual_volume = state.fields.data.shape[0]
    assert actual_volume == expected_volume, (
        f"Foundation Volume Breach: {actual_volume} allocated, "
        f"but math requires {expected_volume} for ghost padding (Rule 9)."
    )

def test_gate_2a_7_point_connectivity_stride():
    """
    Gate 2.A: Topology Connectivity Audit
    
    Identity: id(i+1) = id(c) + stride_x
    Compliance: Rule 7 (Deterministic Indexing via get_flat_index)
    """

    # 1. Setup: Initialize full topology (Step 1 -> Step 2)
    nx, ny, nz = 4, 4, 4
    solver_input = create_validated_input(nx=nx, ny=ny, nz=nz)
    solver_input.grid.nx = nx
    solver_input.grid.ny = ny
    solver_input.grid.nz = nz
    
    context = SimulationContext(input_data=solver_input, config=None)
    
    state = orchestrate_step1(context)
    # Step 2: Assemble the Registry
    stencils = assemble_stencil_matrix(state)
    
    # 2. Define Atomic Truth Stride (Rule 7)
    # In a C-ordered (z, y, x) array, x is the contiguous axis.
    stride_x = 1 
    
    # 3. Grab a block from the logical core (First physical cell)
    block = stencils[0]
    
    # 4. Verification: Check connectivity pointers in the Logic Layer
    center_idx = block.center.index
    ip_idx = block.i_plus.index
    im_idx = block.i_minus.index
    
    assert ip_idx == center_idx + stride_x, (
        f"Topology Breach [X+]: Center index {center_idx} to i_plus {ip_idx} "
        f"does not follow stride {stride_x}."
    )
    assert im_idx == center_idx - stride_x, (
        f"Topology Breach [X-]: Center index {center_idx} to i_minus {im_idx} "
        f"does not follow stride -{stride_x}."
    )

def test_gate_2a_y_stride_integrity():
    """
    Gate 2.A Supplemental: Verify Y-axis stride logic
    
    Identity: id(j+1) = id(c) + (nx+2)
    Compliance: Rule 0 (Data for Logic, Arrays for Math)
    """

    nx, ny, nz = 4, 4, 4
    solver_input = create_validated_input(nx=nx, ny=ny, nz=nz)
    solver_input.grid.nx = nx
    solver_input.grid.ny = ny
    solver_input.grid.nz = nz
    
    context = SimulationContext(input_data=solver_input, config=None)
    
    state = orchestrate_step1(context)
    stencils = assemble_stencil_matrix(state)
    
    # 1. Calculate the Y-stride based on the padded width (Rule 9)
    nx_total = state.grid.nx + 2
    stride_y = nx_total
    
    # 2. Grab an interior block from the Stencil Matrix
    block = stencils[0]
    
    center_idx = block.center.index
    jp_idx = block.j_plus.index
    jm_idx = block.j_minus.index
    
    # 3. Verification: Ensure Y-axis jumps the full width of the padded row
    assert jp_idx == center_idx + stride_y, (
        f"Topology Breach [Y+]: Center index {center_idx} to j_plus {jp_idx} "
        f"expected stride {stride_y} (nx+2)."
    )
    assert jm_idx == center_idx - stride_y, (
        f"Topology Breach [Y-]: Expected stride -{stride_y}."
    )

def test_gate_2a_z_stride_integrity():
    """
    Gate 2.A Supplemental: Verify Z-axis stride logic
    
    Identity: id(k+1) = id(c) + (nx+2)*(ny+2)
    """
    nx, ny, nz = 4, 4, 4
    solver_input = create_validated_input(nx=nx, ny=ny, nz=nz)
    solver_input.grid.nx = nx
    solver_input.grid.ny = ny
    solver_input.grid.nz = nz
    
    context = SimulationContext(input_data=solver_input, config=None)
    
    state = orchestrate_step1(context)
    stencils = assemble_stencil_matrix(state)
    
    # Calculate Z-stride: Jump a full 2D padded slice
    nx_total = state.grid.nx + 2
    ny_total = state.grid.ny + 2
    stride_z = nx_total * ny_total
    
    block = stencils[0]
    
    center_idx = block.center.index
    kp_idx = block.k_plus.index
    
    assert kp_idx == center_idx + stride_z, (
        f"Topology Breach [Z+]: Center index {center_idx} to k_plus {kp_idx} "
        f"expected stride {stride_z} (nx+2 * ny+2)."
    )