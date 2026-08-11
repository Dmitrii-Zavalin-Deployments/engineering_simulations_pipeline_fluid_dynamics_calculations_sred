"""
src/ingestion.py
Strict Schema & Configuration Ingestion Module.
Parses input and configuration JSON files, performing deterministic validation 
against required non-default schemas before passing data into the execution pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

logger = logging.getLogger("Solver.Ingestion")

REQUIRED_INPUT_SECTIONS = [
    "grid",
    "fluid_properties",
    "initial_conditions",
    "simulation_parameters",
    "boundary_conditions",
    "mask",
    "external_forces",
    "domain_configuration",
    "physical_constraints",
]

REQUIRED_GRID_KEYS = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz"]
REQUIRED_FLUID_KEYS = ["density", "viscosity"]
REQUIRED_IC_KEYS = ["velocity", "pressure"]
REQUIRED_SIM_KEYS = ["time_step", "total_time", "output_interval"]
REQUIRED_FORCES_KEYS = ["force_vector", "gravity_vector"]
REQUIRED_CONSTRAINTS_KEYS = ["min_velocity", "max_velocity", "min_pressure", "max_pressure"]
REQUIRED_CONFIG_KEYS = ["max_poisson_iterations", "poisson_tolerance"]


def _validate_keys(data: Dict[str, Any], required_keys: list, section_name: str) -> None:
    for key in required_keys:
        if key not in data or data[key] is None:
            raise KeyError(
                f"Non-default policy violation in '{section_name}': missing required key '{key}'."
            )


def load_and_validate_inputs(input_path: str, config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Reads, parses, and strictly validates the simulation input JSON and configuration JSON files.
    
    Args:
        input_path: Path to navier_stokes_input.json
        config_path: Path to config.json
        
    Returns:
        Tuple containing (input_data_dict, config_data_dict)
    """
    input_file = Path(input_path)
    config_file = Path(config_path)

    if not input_file.is_file():
        raise FileNotFoundError(f"Simulation input file not found at: {input_path}")
    if not config_file.is_file():
        raise FileNotFoundError(f"Solver configuration file not found at: {config_path}")

    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    logger.info("Performing strict schema validation on input JSON...")
    
    # Validate main input sections
    _validate_keys(input_data, REQUIRED_INPUT_SECTIONS, "root_input")

    # Validate individual subsections
    _validate_keys(input_data["grid"], REQUIRED_GRID_KEYS, "grid")
    _validate_keys(input_data["fluid_properties"], REQUIRED_FLUID_KEYS, "fluid_properties")
    _validate_keys(input_data["initial_conditions"], REQUIRED_IC_KEYS, "initial_conditions")
    _validate_keys(input_data["simulation_parameters"], REQUIRED_SIM_KEYS, "simulation_parameters")
    _validate_keys(input_data["external_forces"], REQUIRED_FORCES_KEYS, "external_forces")
    _validate_keys(input_data["physical_constraints"], REQUIRED_CONSTRAINTS_KEYS, "physical_constraints")

    if "type" not in input_data["domain_configuration"]:
        raise KeyError("Non-default policy violation in 'domain_configuration': missing key 'type'.")

    # Validate grid bounds and cell counts
    grid = input_data["grid"]
    if grid["nx"] <= 0 or grid["ny"] <= 0 or grid["nz"] <= 0:
        raise ValueError("Grid cell dimensions (nx, ny, nz) must be strictly positive integers.")
    if grid["x_max"] <= grid["x_min"] or grid["y_max"] <= grid["y_min"] or grid["z_max"] <= grid["z_min"]:
        raise ValueError("Grid physical maximum boundaries must be strictly greater than minimum boundaries.")

    # Validate mask length matches grid discretization
    expected_mask_len = grid["nx"] * grid["ny"] * grid["nz"]
    if len(input_data["mask"]) != expected_mask_len:
        raise ValueError(
            f"Mask length mismatch: expected {expected_mask_len} elements, "
            f"got {len(input_data['mask'])}."
        )

    # Validate configuration parameters
    logger.info("Performing strict schema validation on configuration JSON...")
    _validate_keys(config_data, REQUIRED_CONFIG_KEYS, "config")

    if config_data["max_poisson_iterations"] <= 0:
        raise ValueError("Configuration parameter 'max_poisson_iterations' must be > 0.")
    if config_data["poisson_tolerance"] <= 0.0:
        raise ValueError("Configuration parameter 'poisson_tolerance' must be > 0.0.")

    logger.info("Input and configuration schemas successfully verified.")
    return input_data, config_data
