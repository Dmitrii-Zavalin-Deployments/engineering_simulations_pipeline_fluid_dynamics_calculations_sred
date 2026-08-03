# src/main_solver.py

import json
import logging
import sys
import traceback
from pathlib import Path

import jsonschema
import numpy as np

# Rule 5: Force global arithmetic trapping for deterministic stability
np.seterr(all="raise")

from src.common.archive_service import archive_simulation_artifacts
from src.common.elasticity import ElasticManager
from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4

DEBUG = False
logger = logging.getLogger("Solver.Main")
logger.propagate = True
BASE_DIR = Path(__file__).resolve().parent.parent

def _configure_numerical_runtime(context: SimulationContext):
    """Rule 5: Deterministic Initialization via NumPy error trapping."""
    np.seterr(all="raise", under="ignore")
    logger.info("Numerical runtime configured: Trapping arithmetic anomalies.")

def _load_simulation_context(input_path: str) -> SimulationContext:
    """Assembles physical input and numerical config into a unified context."""
    full_input_path = Path(input_path)
    if not full_input_path.is_absolute():
        full_input_path = BASE_DIR / input_path
        
    config_path = BASE_DIR / "config.json"

    if not full_input_path.exists():
        raise FileNotFoundError(f"Input file missing at {full_input_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.json required at {config_path}")

    with open(full_input_path) as f:
        input_data = json.load(f)
    with open(config_path) as f:
        config_data = json.load(f)

    return SimulationContext.create(input_data, config_data)

def run_solver(input_path: str) -> str:
    """Main Orchestrator with State-Anchored Elastic Stability."""
    context = _load_simulation_context(input_path)
    _configure_numerical_runtime(context)

    # 1. VALIDATE INPUT
    SCHEMA_PATH = BASE_DIR / "schema/solver_input_schema.json"
    try:
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        jsonschema.validate(instance=context.input_data.to_dict(), schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"!!! CONTRACT VIOLATION: {e.message}")
        raise

    # 2. ASSEMBLY
    state = orchestrate_step1(context)
    state = orchestrate_step2(state)

    # 3. STATE CONTRACT VALIDATION
    try:
        state.validate_against_schema(str(SCHEMA_PATH))
    except jsonschema.exceptions.ValidationError as e:
        path_str = ".".join([str(p) for p in e.path])
        logger.error(f"!!! STATE CONTRACT VIOLATION at {path_str}: {e.message}")
        raise

    # 4. ELASTICITY ENGINE
    elasticity = ElasticManager(context.config, state)

    # 5. MAIN EXECUTION LOOP
    while state.ready_for_time_loop:
        # RULE 9: Snapshot memory BEFORE the trial begins
        state.capture_stable_state()

        try:
            # Sync time-step from Elasticity SSoT
            for b in state.stencil_matrix:
                b.dt = elasticity.dt

            # PREDICTOR PASS
            for block in state.stencil_matrix:
                orchestrate_step3(
                    block, 
                    context, 
                    state.grid, 
                    state.boundary_conditions, 
                    is_first_pass=True
                )
            
            # PPE ITERATION
            for _ in range(context.config.ppe_max_iter):
                max_delta = 0.0
                for block in state.stencil_matrix:
                    _, delta = orchestrate_step3(
                        block, 
                        context, 
                        state.grid, 
                        state.boundary_conditions, 
                        is_first_pass=False
                    )
                    max_delta = max(max_delta, delta)

                if max_delta < context.config.ppe_tolerance:
                    break

            # Success signal
            elasticity.stabilization(is_needed=False)

            # Finalize Step (Iteration incremented here)
            state = orchestrate_step4(state, context)

            if state.time >= context.input_data.simulation_parameters.total_time:
                state.ready_for_time_loop = False

        except ArithmeticError as e:
            # RULE 9: ANTI-FRANKENSTEIN ROLLBACK
            # Wipes the memory pollution (e.g., 144.93) before the next dt retry
            logger.error(f"Audit Failure: {e}") 
            
            # This already handles the ghosts because it restores the WHOLE array
            state.rollback_to_stable_state() 
            
            # REQUIRED FOR PYTEST
            logger.warning(f"STABILITY TRIGGER: Physical anomaly at iteration {state.iteration}. Reducing dt...")
            
            # Reduce dt and loop back to try again with clean memory
            elasticity.stabilization(is_needed=True)

        except Exception as e:
            # Catch-all for Terminal errors (ValueError, FloatingPointError, etc.)
            logger.error(f"❌ CRITICAL TERMINATION [{type(e).__name__}]: {e!s}")
            raise

    return archive_simulation_artifacts(state)

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/main_solver.py <input_json_path>")
        sys.exit(1)
    try:
        zip_path = run_solver(sys.argv[1])
        print(f"Pipeline complete. Artifacts archived at: {zip_path}")
        sys.exit(0)
    except Exception as e:
        print(f"FATAL PIPELINE ERROR: {e!s}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":  # pragma: no cover
    main()