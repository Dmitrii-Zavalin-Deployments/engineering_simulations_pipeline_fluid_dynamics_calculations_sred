"""
src/main.py
Main Execution Control Plane.
Orchestrates ingestion, simulation state instantiation, the master time-integration loop,
and archival packaging under strict non-default policies.
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from src.archivist import archive_simulation_results
from src.cpp_gate import step_simulation
from src.ingestion import load_and_validate_inputs
from src.state import SolverState

logger = logging.getLogger("Solver.Main")
BASE_DIR = Path(__file__).resolve().parent.parent


def run_simulation(
    input_output_folder: str | Path,
    input_file_name: str,
    output_file_name: str,
) -> None:
    """
    Executes the complete Navier-Stokes simulation pipeline.

    Args:
        input_output_folder: Directory path containing inputs and receiving output artifacts.
        input_file_name: File name of the input JSON configuration.
        output_file_name: File name for the output JSON manifest.
    """
    if input_output_folder is None:
        raise ValueError("FATAL ERROR: input_output_folder must be explicitly provided (no defaults allowed).")
    if input_file_name is None:
        raise ValueError("FATAL ERROR: input_file_name must be explicitly provided (no defaults allowed).")
    if output_file_name is None:
        raise ValueError("FATAL ERROR: output_file_name must be explicitly provided (no defaults allowed).")

    out_path = Path(input_output_folder).resolve()
    input_path = out_path / input_file_name

    if not input_path.is_file():
        raise FileNotFoundError(f"Input configuration file not found at: {input_path}")

    config_path = BASE_DIR / "config" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    logger.info("Step 1: Loading and validating inputs and configuration...")
    input_data, config_data = load_and_validate_inputs(input_path, config_path)

    logger.info("Step 2: Initializing SolverState container...")
    state = SolverState(input_data, config_data)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting master time-integration loop: {state.total_iterations} iterations, dt={state.dt}")

        # Master time loop
        for _ in range(1, state.total_iterations + 1):
            # Execute physical step through C++ bridge
            step_simulation(state)

            # Enforce physical constraints / bounds / stability checks
            state.enforce_physical_constraints()

            if state.current_iteration % max(1, state.total_iterations // 10) == 0:
                logger.info(
                    f"Progress: Iteration {state.current_iteration}/{state.total_iterations} (t={state.current_time:.4f}s)"
                )

        logger.info("Simulation completed successfully. Packaging results via Archivist...")
        archive_simulation_results(
            state=state,
            output_dir=out_path,
            output_filename=output_file_name,
            status="SUCCESS",
        )
        logger.info("Simulation artifacts successfully archived")

    except Exception as exec_err:
        logger.error(f"Simulation loop encountered unrecoverable failure: {exec_err}")
        logger.info("Attempting to write failure manifest via Archivist...")
        try:
            archive_simulation_results(
                state=state,
                output_dir=out_path,
                output_filename=output_file_name,
                status="FAILURE",
            )
        except Exception as archive_err:
            logger.critical(f"Failed to write failure manifest: {archive_err}")
        raise exec_err


def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="Navier-Stokes Solver Execution Engine")
        parser.add_argument("--input_output_folder", type=str, default=None, help="Directory folder containing input/output artifacts")
        parser.add_argument("--input_file_name", type=str, default=None, help="File name of the input JSON configuration")
        parser.add_argument("--output_file_name", type=str, default=None, help="File name for the output JSON manifest")

        args = parser.parse_args()

        run_simulation(
            input_output_folder=args.input_output_folder,
            input_file_name=args.input_file_name,
            output_file_name=args.output_file_name,
        )
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        AttributeError,
    ) as e:
        print(f"FATAL PIPELINE ERROR: {e!s}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
