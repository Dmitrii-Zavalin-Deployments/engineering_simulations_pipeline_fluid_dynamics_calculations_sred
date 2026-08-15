"""
src/main.py
Main Execution Control Plane.
Orchestrates ingestion, sovereign state instantiation, the master time-integration loop,
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


def run_simulation(input_path: str | Path, output_dir: str | Path, zip_filename: str = "simulation_results.zip") -> str:
    """
    Executes the complete Navier-Stokes simulation pipeline.

    Args:
        input_path: Path to navier_stokes_input.json (mandatory, no defaults)
        output_dir: Target directory path for output artifacts (mandatory, no defaults)
        zip_filename: Name of the generated zip archive (mandatory, no defaults)

    Returns:
        Absolute path string to the generated archive.
    """
    if input_path is None:
        raise ValueError("FATAL ERROR: input_path must be explicitly provided (no defaults allowed).")
    if output_dir is None:
        raise ValueError("FATAL ERROR: output_dir must be explicitly provided (no defaults allowed).")
    if zip_filename is None:
        raise ValueError("FATAL ERROR: zip_filename must be explicitly provided (no defaults allowed).")

    config_path = BASE_DIR / "config" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    logger.info("Step 1: Loading and validating inputs and configuration...")
    input_data, config_data = load_and_validate_inputs(input_path, config_path)

    logger.info("Step 2: Initializing Sovereign SolverState container...")
    state = SolverState(input_data, config_data)

    logger.info(f"Starting master time-integration loop: {state.total_iterations} iterations, dt={state.dt}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

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
    archive_path = archive_simulation_results(state, str(out_path), zip_filename)
    logger.info(f"Simulation artifacts successfully archived at: {archive_path}")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Navier-Stokes Solver Execution Engine")
    parser.add_argument("positional_input", nargs="?", default=None, help="Path to input JSON configuration file")
    parser.add_argument("--input_output_folder", type=str, default=None, help="Directory folder containing input/output artifacts")
    parser.add_argument("--input_file_name", type=str, default=None, help="File name of the input JSON configuration")
    parser.add_argument("--output_file_name", type=str, default=None, help="File name for the output archive zip")

    args = parser.parse_args()

    has_folder = args.input_output_folder is not None
    has_input_file = args.input_file_name is not None
    has_output_file = args.output_file_name is not None
    has_positional = args.positional_input is not None

    if has_folder or has_input_file or has_output_file:
        if not has_folder:
            raise ValueError("FATAL PIPELINE ERROR: --input_output_folder must be explicitly provided.")
        if not has_input_file:
            raise ValueError("FATAL PIPELINE ERROR: --input_file_name must be explicitly provided.")
        if not has_output_file:
            raise ValueError("FATAL PIPELINE ERROR: --output_file_name must be explicitly provided.")

        input_path = Path(args.input_output_folder) / args.input_file_name
        output_dir = Path(args.input_output_folder)
        zip_filename = args.output_file_name
    elif has_positional:
        input_path = Path(args.positional_input)
        output_dir = Path(args.positional_input).parent
        zip_filename = "simulation_results.zip"
    else:
        raise ValueError("FATAL PIPELINE ERROR: Must provide either positional <input_json> or all required flag arguments (--input_output_folder, --input_file_name, --output_file_name).")

    try:
        run_simulation(input_path=input_path, output_dir=output_dir, zip_filename=zip_filename)
        sys.exit(0)
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


if __name__ == "__main__":
    main()
