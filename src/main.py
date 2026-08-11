# src/main.py

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import jsonschema
import numpy as np

# Rule 5: Force global arithmetic trapping for deterministic stability
np.seterr(all="raise")

from src.archivist import archive_simulation_results
from src.ingestion import load_and_validate_inputs
from src.state import SolverState
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4

DEBUG = False
logger = logging.getLogger("Solver.Main")
logger.propagate = True
BASE_DIR = Path(__file__).resolve().parent.parent


class InputDataWrapper:
    """Provides attribute access over raw simulation input dictionaries."""

    def __init__(self, input_data: dict[str, Any]):
        if input_data is None:
            raise ValueError("FATAL ERROR: input_data must be explicitly provided.")
        self._data = input_data
        self.simulation_parameters = input_data.get("simulation_parameters")

    def to_dict(self) -> dict[str, Any]:
        return self._data


class SimulationContext:
    """Container combining input schema parameters and execution configuration."""

    def __init__(self, input_data: dict[str, Any], config: dict[str, Any]):
        if input_data is None or config is None:
            raise ValueError("FATAL ERROR: input_data and config must be explicitly provided.")
        self.input_data = InputDataWrapper(input_data)
        self.config = config

    @classmethod
    def create(cls, input_data: dict[str, Any], config_data: dict[str, Any]) -> "SimulationContext":
        if input_data is None or config_data is None:
            raise ValueError("FATAL ERROR: input_data and config_data must be explicitly provided.")
        return cls(input_data, config_data)


class ElasticManager:
    """Manages solver time-step stability and adaptive scaling under anomalies."""

    def __init__(self, config: dict[str, Any], state: SolverState):
        if config is None or state is None:
            raise ValueError("FATAL ERROR: config and state must be explicitly provided.")
        self.config: dict[str, Any] = config
        self.state: SolverState = state
        self.dt: float = float(state.dt)

    def stabilization(self, is_needed: bool = False) -> None:
        if is_needed is None:
            raise ValueError("FATAL ERROR: is_needed must be explicitly provided.")
        if is_needed:
            self.dt *= 0.5
            logger.warning(f"Elasticity stabilization triggered: reducing time-step dt to {self.dt}")


def archive_simulation_artifacts(state: SolverState, output_path: str | Path) -> str:
    """Routes artifact archival requests to the core archivist."""
    if state is None or output_path is None:
        raise ValueError("FATAL ERROR: state and output_path must be explicitly provided.")
    p = Path(output_path)
    if p.is_dir() or str(output_path).endswith("/") or not p.suffix:
        p.mkdir(parents=True, exist_ok=True)
        return archive_simulation_results(state, str(p), "simulation_results.zip")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        return archive_simulation_results(state, str(p.parent), p.name)


def _configure_numerical_runtime(context: SimulationContext):
    """Rule 5: Deterministic Initialization via NumPy error trapping."""
    np.seterr(all="raise", under="ignore")
    logger.info("Numerical runtime configured: Trapping arithmetic anomalies.")


def _load_simulation_context(input_path: str | Path) -> SimulationContext:
    """Assembles physical input and numerical config into a unified context."""
    if input_path is None:
        raise ValueError("FATAL ERROR: input_path must be explicitly provided (no defaults allowed).")

    full_input_path = Path(input_path)
    if not full_input_path.is_absolute():
        full_input_path = BASE_DIR / input_path

    config_path = BASE_DIR / "config/config.json"

    if not full_input_path.exists():
        raise FileNotFoundError(f"Input file missing at {full_input_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.json required at {config_path}")

    input_data, config_data = load_and_validate_inputs(full_input_path, config_path)
    return SimulationContext.create(input_data, config_data)


def run_solver(input_path: str | Path, output_path: str | Path) -> str:
    """Main Orchestrator with State-Anchored Elastic Stability."""
    if input_path is None:
        raise ValueError("FATAL ERROR: input_path must be explicitly provided.")
    if output_path is None:
        raise ValueError("FATAL ERROR: output_path must be explicitly provided.")

    context = _load_simulation_context(input_path)
    _configure_numerical_runtime(context)

    # 1. VALIDATE INPUT
    SCHEMA_PATH = BASE_DIR / "schema/solver_input_schema.json"
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Solver input schema missing at {SCHEMA_PATH}")

    try:
        with open(SCHEMA_PATH, encoding="utf-8") as f:
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
        logger.error(f"!!! STATE CONTRACT VALIDATION at {path_str}: {e.message}")
        raise

    # 4. ELASTICITY ENGINE
    elasticity = ElasticManager(context.config, state)

    # 5. MAIN EXECUTION LOOP
    while state.ready_for_time_loop:
        state.capture_stable_state()

        try:
            for b in state.stencil_matrix:
                b.dt = elasticity.dt

            for block in state.stencil_matrix:
                orchestrate_step3(
                    block,
                    context,
                    state.grid,
                    state.boundary_conditions,
                    is_first_pass=True,
                )

            for _ in range(context.config["max_poisson_iterations"]):
                max_delta = 0.0
                for block in state.stencil_matrix:
                    _, delta = orchestrate_step3(
                        block,
                        context,
                        state.grid,
                        state.boundary_conditions,
                        is_first_pass=False,
                    )
                    max_delta = max(max_delta, delta)

                if max_delta < context.config["poisson_tolerance"]:
                    break

            elasticity.stabilization(is_needed=False)
            state = orchestrate_step4(state, context)

            if state.time >= context.input_data.simulation_parameters["total_time"]:
                state.ready_for_time_loop = False

        except ArithmeticError as e:
            logger.error(f"Audit Failure: {e}")
            state.rollback_to_stable_state()
            logger.warning(
                f"STABILITY TRIGGER: Physical anomaly at iteration {state.iteration}. Reducing dt..."
            )
            elasticity.stabilization(is_needed=True)

        except (RuntimeError, TypeError, ValueError, AttributeError) as e:
            logger.error(f"CRITICAL TERMINATION [{type(e).__name__}]: {e!s}")
            raise

    return archive_simulation_artifacts(state, output_path=output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Navier–Stokes Solver Execution Engine"
    )
    parser.add_argument(
        "positional_input",
        nargs="?",
        default=None,
        help="Path to input JSON configuration file (positional argument)",
    )
    parser.add_argument(
        "--input_output_folder",
        type=str,
        default=None,
        help="Directory folder containing input/output artifacts",
    )
    parser.add_argument(
        "--input_file_name",
        type=str,
        default=None,
        help="File name of the input JSON configuration",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default=None,
        help="File name for the output archive zip",
    )

    args = parser.parse_args()

    if args.input_output_folder and args.input_file_name:
        input_path = Path(args.input_output_folder) / args.input_file_name
    elif args.positional_input:
        input_path = Path(args.positional_input)
    else:
        raise ValueError(
            "FATAL PIPELINE ERROR: Must provide either positional <input_json_path> OR both --input_output_folder and --input_file_name"
        )

    if args.input_output_folder and args.output_file_name:
        output_path = Path(args.input_output_folder) / args.output_file_name
    elif args.output_file_name:
        output_path = Path(args.output_file_name)
    elif args.positional_input:
        output_path = Path(args.positional_input).parent / "simulation_results.zip"
    else:
        raise ValueError(
            "FATAL PIPELINE ERROR: Output path must be explicitly provided or derivable from input."
        )

    try:
        zip_path = run_solver(input_path=input_path, output_path=output_path)
        print(f"Pipeline complete. Artifacts archived at: {zip_path}")
        sys.exit(0)
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        jsonschema.exceptions.ValidationError,
        ArithmeticError,
        OSError,
        json.JSONDecodeError,
        RuntimeError,
        TypeError,
        AttributeError,
    ) as e:
        print(f"FATAL PIPELINE ERROR: {e!s}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
