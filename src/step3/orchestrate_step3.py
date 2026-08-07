# src/step3/orchestrate_step3.py

from src.common.simulation_context import SimulationContext
from src.common.stencil_block import StencilBlock

# Rule 8: Granular Sub-module Access
from src.step3.boundaries.applier import apply_boundary_values
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs
from src.step3.corrector import apply_local_velocity_correction
from src.step3.ops.ghost_handler import sync_ghost_trial_buffers
from src.step3.ppe_solver import solve_pressure_poisson_step
from src.step3.predictor import compute_local_predictor_step

# Rule 7: Granular Traceability for GitHub Actions
DEBUG = False

def orchestrate_step3(
    block: StencilBlock, 
    context: SimulationContext, 
    state_grid: object,
    state_bc_manager: object,
    is_first_pass: bool = False
) -> tuple[StencilBlock, float]:
    """
    Step 3 Orchestrator: Eulerian Mask-Filtered Projection Solver.
    
    Execution Flow per Cell Mask:
    - Ghost / Boundary (is_ghost or mask == -1): Enforce BC schema and sync ghost buffers.
    - Solid Cell (mask == 0): Early return / skip physics to prevent vacuum sinks.
    - Fluid Cell (mask == 1): Execute Predictor (u*), PPE Divergence/Poisson Solve, and Corrector (u^{n+1}).
    """
    
    # =========================================================================
    # PHASE 1: GHOST & DOMAIN BOUNDARY HANDLING (mask == -1 or is_ghost)
    # =========================================================================
    
    # 1A. Ghost Buffer Synchronization
    if block.center.is_ghost:
        sync_ghost_trial_buffers(block)
        return block, 0.0

    # Retrieve boundary configurations for boundary cells
    rules = get_applicable_boundary_configs(
        block, 
        state_bc_manager.to_dict(),
        state_grid, 
        context.input_data.domain_configuration.to_dict()
    )

    # 1B. Boundary Mask Check (-1)
    mask = getattr(block.center, "mask", 1)
    if mask == -1 or getattr(block.center, "is_boundary", False):
        for rule in rules:
            apply_boundary_values(block, rule)
        return block, 0.0

    # =========================================================================
    # PHASE 2: INTERNAL SOLID CELL FILTERING (mask == 0)
    # =========================================================================
    # Bypasses internal solid cells to prevent garbage leakage and vacuum sinks
    if mask == 0 or getattr(block.center, "is_solid", False):
        return block, 0.0

    # =========================================================================
    # PHASE 3: ACTIVE FLUID CELL PHYSICS (mask == 1)
    # =========================================================================
    if mask == 1 or getattr(block.center, "is_fluid", True):
        
        # --- STEP 1: PREDICTOR STEP (Calculate Intermediate u*) ---
        if is_first_pass:
            compute_local_predictor_step(block)
            return block, 0.0

        # --- STEP 2 & 3: PRESSURE POISSON SOLVE (Divergence Evaluation & PPE) ---
        delta = solve_pressure_poisson_step(
            block, 
            context.config.divergence_threshold, 
            context.config.ppe_omega
        )

        # --- STEP 4: CORRECTOR STEP (Project Field to Divergence-Free Space) ---
        apply_local_velocity_correction(block)

        return block, delta

    return block, 0.0
