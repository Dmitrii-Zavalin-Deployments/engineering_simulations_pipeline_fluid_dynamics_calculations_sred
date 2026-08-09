# tests/property_integrity/test_architecture_parity.py

import pytest

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

# Dummies updated to match the orchestration steps in main.py
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# --- ARCHITECTURE CATEGORIZATION ---

# Stages using the monolithic SolverState.fields manager (per src/main.py)
STATE_BASED_STAGES = [
    ("Step 1 (Init)", make_step1_output_dummy),
    ("Step 2 (Assembly)", make_step2_output_dummy),
    ("Step 4 (Archivist)", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

# Stages using component-based StencilBlocks (Inside the Step 3 loop)
BLOCK_BASED_STAGES = [
    ("Step 3 (PPE/Predictor)", make_step3_output_dummy),
]

ALL_STAGES = STATE_BASED_STAGES + BLOCK_BASED_STAGES

# --- ARCHITECTURE BRIDGE HELPERS ---

def get_fluid_param(obj, param_name):
    """Extracts physics constants regardless of the container type."""
    if isinstance(obj, StencilBlock):
        # StencilBlocks use direct private attributes for density/viscosity
        mapping = {"density": "_rho", "viscosity": "_mu"}
        return getattr(obj, mapping[param_name], None)
    
    # SolverState uses the _fluid_properties object
    fluid = getattr(obj, "_fluid_properties", None)
    if fluid:
        attr = f"_{param_name}"
        return getattr(fluid, attr, None)
    return None

def get_bc_list(obj):
    """Extracts boundary conditions list from either State or Block."""
    if isinstance(obj, StencilBlock):
        return getattr(obj, "_bc_list", [])
    
    manager = getattr(obj, "_boundary_conditions", None)
    if manager:
        return getattr(manager, "_conditions", [])
    return []

# --- MEMORY & ALLOCATION TESTS ---

@pytest.mark.parametrize("stage_name, factory", STATE_BASED_STAGES)
def test_lifecycle_grid_dimensions_match_fields(stage_name, factory):
    """Robustness: Verifies monolithic buffer size matches (nx+2)*(ny+2)*(nz+2)."""
    nx, ny, nz = 8, 6, 4
    n_expected = (nx + 2) * (ny + 2) * (nz + 2)
    state = factory(nx=nx, ny=ny, nz=nz)
    
    # Verify the monolithic fields.data array
    data = state.fields.data
    for field_idx in [FI.P, FI.VX, FI.VY, FI.VZ]:
        actual_size = data[:, field_idx].size
        assert actual_size == n_expected, (
            f"{stage_name}: Field {field_idx} size mismatch. "
            f"Expected {n_expected}, got {actual_size}"
        )

@pytest.mark.parametrize("stage_name, factory", BLOCK_BASED_STAGES)
def test_block_allocation_integrity(stage_name, factory):
    """
    Validation: Verify StencilBlocks are wired to the global Foundation buffer (Rule 9).
    In Step 3, Cell attributes act as pointers/views into the monolithic array.
    """
    import numpy as np

    from src.common.field_schema import FI
    
    nx, ny, nz = 5, 5, 5
    # Standard ghost-cell padding (+2) results in (5+2)^3 = 343 cells
    n_expected_cells = (nx + 2) * (ny + 2) * (nz + 2)
    n_total_elements = n_expected_cells * FI.num_fields()
    
    # Initialize the block using the provided factory
    block = factory(nx=nx, ny=ny, nz=nz)
    
    # Audit each physical field for wiring integrity
    for attr in ["vx", "vy", "vz", "p"]:
        # 1. Verification of existence
        val = getattr(block.center, attr)
        assert val is not None, f"{stage_name}: Component {attr} is None"

        # 2. GATE 1: Type Integrity
        # We allow float or numpy types, provided they are effectively scalars.
        assert isinstance(val, (float, np.float64, np.ndarray)), (
            f"{stage_name}: {attr} returned unexpected type {type(val)}."
        )

        # 3. GATE 2: The "Foundation" Identity Check (Rule 9 - Structural Persistence)
        # We verify that the 'Wiring' (Object) correctly manipulates the 'Foundation' (Buffer).
        test_signature = 123.456789
        field_id = FI[attr.upper()]

        # Action: Set value via the Object's explicit setter
        # Example: block.center.set_vx(123.456)
        setter_name = attr.lower()
        # Property alignment
        setattr(block.center, setter_name, test_signature)

        # Validation: Access the global buffer directly through the factory/manager
        # This confirms that the setter wrote to the SHARED memory, not a local copy.
        # Note: 'block.fields_buffer' or similar access depends on your specific container structure
        actual_buffer_val = block.center.fields_buffer[block.center.index, field_id]

        assert np.isclose(actual_buffer_val, test_signature), (
            f"RULE 9 CRITICAL VIOLATION in {stage_name}: {attr} is a detached copy. "
            f"Object wrote {test_signature} to index {block.center.index}, "
            f"but Foundation Buffer contains {actual_buffer_val}. "
            "Memory MUST be a view of the global buffer to ensure spatial coupling."
        )

        # 4. GATE 3: Global Buffer Scaling Audit
        # Ensure the underlying memory allocation matches the expected grid dimensions.
        actual_buffer_size = block.center.fields_buffer.size
        assert actual_buffer_size == n_total_elements, (
            f"{stage_name}: {attr} wiring mismatch. "
            f"Expected global buffer of {n_total_elements} elements, "
            f"but detected {actual_buffer_size}."
        )

# --- PHYSICS & BOUNDARY PERSISTENCE ---

@pytest.mark.parametrize("stage_name, factory", ALL_STAGES)
def test_fluid_constants_persistence(stage_name, factory):
    """Physics: Ensure density and viscosity are positive and reachable in all stages."""
    obj = factory()
    rho = get_fluid_param(obj, "density")
    mu = get_fluid_param(obj, "viscosity")
    
    assert rho is not None and rho > 0, f"{stage_name}: Missing or non-physical density: {rho}"
    assert mu is not None and mu > 0, f"{stage_name}: Missing or non-physical viscosity: {mu}"

def test_staggered_component_schema_validity():
    """Safety: Verify BC values strictly follow the component schema: {u, v, w, p}."""
    # We check Step 1 as it is the entry point for BC definition
    state = make_step1_output_dummy()
    allowed_keys = {"u", "v", "w", "p"}
    bcs = get_bc_list(state)
    
    assert len(bcs) > 0, "Test Setup Error: Dummy state has no boundary conditions to audit."
    
    for bc in bcs:
        provided_keys = set(getattr(bc, "_values", {}).keys())
        # Ensure no illegal keys (like 'velocity_x') leaked into the dictionary
        assert provided_keys.issubset(allowed_keys), (
            f"Illegal component in BC: {provided_keys - allowed_keys}. "
            f"Must be subset of {allowed_keys}"
        )