"""
src/main.py
Main Execution Control Plane.
Orchestrates ingestion, sovereign state instantiation, the master time-integration loop,
preview generation, and archival packaging.
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from src.archivist import archive_simulation_results
from src.cpp_gate import step_simulation
from src.generate_previews import generate_snapshot_preview
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

    # Record initial state snapshot (iteration 0)
    preview_file = None
    if state.output_interval > 0 and state.current_iteration % state.output_interval == 0:
        preview_file = generate_snapshot_preview(state, out_path, state.current_iteration)
    state.record_snapshot(preview_file_path=preview_file)

    # Master time loop
    for _ in range(1, state.total_iterations + 1):
        # Execute physical step through C++ bridge
        step_simulation(state)

        # Enforce physical constraints / bounds / stability checks
        state.enforce_physical_constraints()

        # Generate preview and record snapshot at intervals
        preview_file = None
        if state.output_interval > 0 and (
            state.current_iteration % state.output_interval == 0
            or state.current_iteration == state.total_iterations
        ):
            preview_file = generate_snapshot_preview(state, out_path, state.current_iteration)

        state.record_snapshot(preview_file_path=preview_file)

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

    if args.input_output_folder and args.input_file_name:
        input_path = Path(args.input_output_folder) / args.input_file_name
    elif args.positional_input:
        input_path = Path(args.positional_input)
    else:
        raise ValueError("FATAL PIPELINE ERROR: Must provide either positional <input_json> or both --input_output_folder and --input_file_name.")

    if args.input_output_folder and args.output_file_name:
        output_dir = Path(args.input_output_folder)
        zip_filename = args.output_file_name
    elif args.output_file_name:
        output_dir = Path(args.output_file_name).parent
        zip_filename = Path(args.output_file_name).name
        if not str(output_dir) or str(output_dir) == ".":
            output_dir = Path(".")
    elif args.positional_input:
        output_dir = Path(args.positional_input).parent
        zip_filename = "simulation_results.zip"
    else:
        raise ValueError("FATAL PIPELINE ERROR: Output path must be explicitly provided or derivable from input.")

    try:
        run_simulation(input_path=input_path, output_dir=output_dir, zip_filename=zip_filename)
        sys.exit(0)
    except Exception as e:
        print(f"FATAL PIPELINE ERROR: {e!s}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
