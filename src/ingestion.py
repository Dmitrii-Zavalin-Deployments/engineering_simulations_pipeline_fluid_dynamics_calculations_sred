"""
src/ingestion.py
Strict Schema & Configuration Ingestion Module.
Parses input and configuration JSON files, performing deterministic validation 
via the official jsonschema package against formal JSON schema definitions.
"""

import json
import logging
from pathlib import Path
from typing import Any

import jsonschema

logger = logging.getLogger("Solver.Ingestion")

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_DIR = BASE_DIR / "schema"


def load_and_validate_inputs(
    input_path: str | Path, config_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Reads, parses, and strictly validates simulation input and configuration 
    against formal JSON schemas using the jsonschema library.
    
    Args:
        input_path: Path to navier_stokes_input.json (mandatory, no defaults)
        config_path: Path to config.json (mandatory, no defaults)
        
    Returns:
        Tuple containing (input_data_dict, config_data_dict)
    """
    if input_path is None:
        raise ValueError("FATAL ERROR: input_path must be explicitly provided (no defaults allowed).")
    if config_path is None:
        raise ValueError("FATAL ERROR: config_path must be explicitly provided (no defaults allowed).")

    input_file = Path(input_path)
    config_file = Path(config_path)

    if not input_file.is_file():
        raise FileNotFoundError(f"Simulation input file not found at: {input_path}")
    if not config_file.is_file():
        raise FileNotFoundError(f"Solver configuration file not found at: {config_file}")

    # Load JSON files
    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # Load corresponding JSON schemas
    input_schema_path = SCHEMA_DIR / "solver_input_schema.json"
    config_schema_path = SCHEMA_DIR / "solver_config_schema.json"

    if not input_schema_path.is_file():
        raise FileNotFoundError(f"Input schema not found at: {input_schema_path}")
    if not config_schema_path.is_file():
        raise FileNotFoundError(f"Config schema not found at: {config_schema_path}")

    with open(input_schema_path, "r", encoding="utf-8") as f:
        input_schema = json.load(f)

    with open(config_schema_path, "r", encoding="utf-8") as f:
        config_schema = json.load(f)

    logger.info("Performing formal schema validation on input JSON via jsonschema...")
    try:
        jsonschema.validate(instance=input_data, schema=input_schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Input schema validation failed: {e.message} at path {'/'.join(str(p) for p in e.path)}")

    logger.info("Performing formal schema validation on configuration JSON via jsonschema...")
    try:
        jsonschema.validate(instance=config_data, schema=config_schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Config schema validation failed: {e.message} at path {'/'.join(str(p) for p in e.path)}")

    # --- Cross-Field & Semantic Validations (Not easily expressible purely in standard JSON Schema) ---
    grid = input_data["grid"]
    if grid["x_max"] <= grid["x_min"] or grid["y_max"] <= grid["y_min"] or grid["z_max"] <= grid["z_min"]:
        raise ValueError("Grid physical maximum boundaries must be strictly greater than minimum boundaries.")

    expected_mask_len = grid["nx"] * grid["ny"] * grid["nz"]
    if len(input_data["mask"]) != expected_mask_len:
        raise ValueError(
            f"Mask length mismatch: expected {expected_mask_len} elements, "
            f"got {len(input_data['mask'])}."
        )

    logger.info("Input and configuration schemas successfully verified.")
    return input_data, config_data