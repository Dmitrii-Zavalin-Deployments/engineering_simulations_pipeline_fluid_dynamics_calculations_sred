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
    Step 3 Orchestrator: Stabilized Collocated Projection Method.
    
    Mathematical Flow (Collocated Rhie-Chow):
    1. Predictor: Compute v* (intermediate velocity).
    2. PPE Loop (Iterative):
       a. Solve Pressure (SOR) with Rhie-Chow Laplacian correction.
       b. Correct v* immediately using new pressure gradient.
       c. Re-enforce Boundary Conditions on v* to ensure wall-consistency.
    """
    
    # --- [A] GHOST SYNC PATH ---
    # Rule 9: Short-circuit ghosts to maintain data foundation integrity
    if block.center.is_ghost:
        sync_ghost_trial_buffers(block)
        return block, 0.0
    
    # --- [B] SHARED BOUNDARY DISPATCHER ---
    # Boundary rules are needed in both Predictor and Iterative phases
    rules = get_applicable_boundary_configs(
        block, 
        state_bc_manager.to_dict(),
        state_grid, 
        context.input_data.domain_configuration.to_dict()
    )
    
    # --- [C] PHYSICS KERNEL PATH ---
    
    # PHASE 1: PREDICT (Once per dt)
    if is_first_pass:
        # A. Intermediate star-velocity calculation (v*)
        compute_local_predictor_step(block)
        
        # B. Initial Boundary Enforcement
        # Ensures that the first PPE iteration sees correct Inlet/Outlet/Wall values
        for rule in rules:
            apply_boundary_values(block, rule)
            
        return block, 0.0

    # PHASE 2: SOLVE & CORRECT (Iterative)
    # 1. Solve: Pressure Poisson Equation (SOR)
    # Mapping Rule: (block, divergence_threshold, ppe_omega)
    # Decouples ppe_solver from StencilBlock internal configuration hierarchy.
    delta = solve_pressure_poisson_step(
        block, 
        context.config.divergence_threshold, 
        context.config.ppe_omega
    )

    # 2. Correct: Velocity Projection
    # This projects v* onto the divergence-free space defined by p^{n+1}
    # Because we write to VX_STAR, the next iteration's div_v_star is updated.
    apply_local_velocity_correction(block)

    # 3. Re-Enforce: Boundary Consistency
    # CRITICAL: Velocity correction can "drift" values at the boundaries.
    # We must reset No-Slip or Inlet conditions before the next SOR iteration.
    for rule in rules:
        apply_boundary_values(block, rule)

    return block, delta