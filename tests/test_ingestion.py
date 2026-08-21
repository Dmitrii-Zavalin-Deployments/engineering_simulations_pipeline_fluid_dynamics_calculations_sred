"""
@file test_ingestion.py
@brief Literate Test Suite for Strict Schema & Configuration Ingestion Module (src/ingestion.py)

This test module serves as a narrative document and exhaustive verification suite for src/ingestion.py.
Explanatory text and physical discretization formulas are written as commented prose, while
executable Python assertions verify strict non-default schema parsing via jsonschema,
domain spatial boundary checks, mask length alignment, and exception handling across all failure modes.
"""

import json
from pathlib import Path

import pytest

from src.ingestion import load_and_validate_inputs


def _write_json(path: Path, data: dict) -> Path:
    """Helper utility to serialize dictionary payloads to disk for test execution."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def _create_valid_schemas():
    """Returns valid minimal input and configuration dictionary structures."""
    valid_input = {
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "z_min": 0.0,
            "z_max": 1.0,
            "nx": 2,
            "ny": 2,
            "nz": 2,
        },
        "fluid_properties": {"density": 1000.0, "viscosity": 0.001},
        "initial_conditions": {"velocity": [0.0, 0.0, 0.0], "pressure": 0.0},
        "simulation_parameters": {
            "time_step": 0.001,
            "total_time": 0.1,
            "output_interval": 10,
        },
        "boundary_conditions": [],
        "mask": [1] * 8,  # Total elements = nx * ny * nz = 2 * 2 * 2 = 8
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "gravity_vector": [0.0, -9.81, 0.0],
        },
        "domain_configuration": {"type": "INTERNAL"},
        "physical_constraints": {
            "min_velocity": -10.0,
            "max_velocity": 10.0,
            "min_pressure": -100.0,
            "max_pressure": 100.0,
        },
    }

    valid_config = {
        "max_poisson_iterations": 50,
        "poisson_tolerance": 1e-6,
    }

    return valid_input, valid_config


# ============================================================================
# NARRATIVE SECTION 1: Successful Schema Verification & Storage Ingestion
# ============================================================================
# Numerical spatial domain discretization requires a structured 3D tensor grid:
#     N_total = nx * ny * nz
#
# Spatial coordinates must satisfy strict physical inequality bounds:
#     x_min < x_max,  y_min < y_max,  z_min < z_max
# ============================================================================


def test_load_and_validate_inputs_success(tmp_path):
    """Verifies successful loading and validation of compliant simulation input and config files."""
    valid_input, valid_config = _create_valid_schemas()

    input_file = _write_json(tmp_path / "navier_stokes_input.json", valid_input)
    config_file = _write_json(tmp_path / "config.json", valid_config)

    # We execute schema ingestion:
    loaded_input, loaded_config = load_and_validate_inputs(input_file, config_file)

    # Total grid elements assertion:
    nx, ny, nz = loaded_input["grid"]["nx"], loaded_input["grid"]["ny"], loaded_input["grid"]["nz"]
    total_cells = nx * ny * nz
    assert total_cells == 8
    assert len(loaded_input["mask"]) == total_cells

    # Assert retrieved contents match source JSON dictionaries:
    assert loaded_input["grid"]["x_max"] > loaded_input["grid"]["x_min"]
    assert loaded_config["max_poisson_iterations"] == 50


# ============================================================================
# NARRATIVE SECTION 2: Mandatory Argument Path Validation
# ============================================================================
# The ingestion system prohibits default arguments. Both input_path and
# config_path must be explicitly provided non-null string or Path references:
#     input_path != None  and  config_path != None
# ============================================================================


def test_load_and_validate_inputs_null_paths(tmp_path):
    """Verifies that passing None to input_path or config_path raises fatal ValueErrors."""
    valid_input, valid_config = _create_valid_schemas()
    config_file = _write_json(tmp_path / "config.json", valid_config)
    input_file = _write_json(tmp_path / "input.json", valid_input)

    # We assert null input_path raises ValueError:
    with pytest.raises(ValueError, match="input_path must be explicitly provided"):
        load_and_validate_inputs(None, config_file)

    # We assert null config_path raises ValueError:
    with pytest.raises(ValueError, match="config_path must be explicitly provided"):
        load_and_validate_inputs(input_file, None)


def test_load_and_validate_inputs_missing_files(tmp_path):
    """Verifies FileNotFoundError when input or configuration files do not exist on disk."""
    valid_input, valid_config = _create_valid_schemas()
    input_file = _write_json(tmp_path / "input.json", valid_input)
    config_file = _write_json(tmp_path / "config.json", valid_config)

    missing_path = tmp_path / "non_existent.json"

    # Assert non-existent input file raises FileNotFoundError:
    with pytest.raises(FileNotFoundError, match="Simulation input file not found"):
        load_and_validate_inputs(missing_path, config_file)

    # Assert non-existent config file raises FileNotFoundError:
    with pytest.raises(FileNotFoundError, match="Solver configuration file not found"):
        load_and_validate_inputs(input_file, missing_path)


# ============================================================================
# NARRATIVE SECTION 3: Formal JSON Schema Compliance Enforcement
# ============================================================================
# Every required schema key and type constraint is validated via jsonschema.
# Missing fields or schema violations trigger a ValueError.
# ============================================================================


def test_schema_missing_required_keys(tmp_path):
    """Verifies ValueError is raised when required schema sections are missing."""
    valid_input, valid_config = _create_valid_schemas()
    config_file = _write_json(tmp_path / "config.json", valid_config)

    # Remove required section 'fluid_properties'
    del valid_input["fluid_properties"]
    input_file = _write_json(tmp_path / "input_invalid.json", valid_input)

    with pytest.raises(ValueError, match="Input schema validation failed"):
        load_and_validate_inputs(input_file, config_file)


def test_domain_configuration_type_missing_or_invalid(tmp_path):
    """Verifies ValueError when domain_configuration section lacks 'type' or has invalid enum value."""
    valid_input, valid_config = _create_valid_schemas()
    config_file = _write_json(tmp_path / "config.json", valid_config)

    # Case A: Missing 'type' key
    valid_input["domain_configuration"] = {}
    input_file_a = _write_json(tmp_path / "input_a.json", valid_input)

    with pytest.raises(ValueError, match="Input schema validation failed"):
        load_and_validate_inputs(input_file_a, config_file)

    # Case B: Invalid enum value ('3d_cube' is not in ['INTERNAL', 'EXTERNAL'])
    valid_input["domain_configuration"] = {"type": "3d_cube"}
    input_file_b = _write_json(tmp_path / "input_b.json", valid_input)

    with pytest.raises(ValueError, match="Input schema validation failed"):
        load_and_validate_inputs(input_file_b, config_file)


# ============================================================================
# NARRATIVE SECTION 4: Grid Discretization Bounds & Spatial Consistency
# ============================================================================
# Discretization resolution components (nx, ny, nz) must be strictly positive integers:
#     nx > 0,  ny > 0,  nz > 0
#
# Spatial extents require positive non-zero spatial lengths:
#     dx = (x_max - x_min) > 0
#     dy = (y_max - y_min) > 0
#     dz = (z_max - z_min) > 0
# ============================================================================


def test_grid_cell_dimensions_non_positive(tmp_path):
    """Verifies ValueError when grid cell dimensions (nx, ny, nz) violate schema bounds (<= 0)."""
    valid_input, valid_config = _create_valid_schemas()
    config_file = _write_json(tmp_path / "config.json", valid_config)

    # Set non-positive nx dimension:
    valid_input["grid"]["nx"] = 0
    input_file = _write_json(tmp_path / "input.json", valid_input)

    with pytest.raises(ValueError, match="Input schema validation failed"):
        load_and_validate_inputs(input_file, config_file)


def test_grid_physical_boundary_limits_invalid(tmp_path):
    """Verifies ValueError when maximum spatial boundaries are <= minimum boundaries."""
    valid_input, valid_config = _create_valid_schemas()
    config_file = _write_json(tmp_path / "config.json", valid_config)

    # Set invalid x bounds where x_max <= x_min:
    valid_input["grid"]["x_min"] = 1.0
    valid_input["grid"]["x_max"] = 1.0  # dx = 1.0 - 1.0 = 0.0 (Invalid)
    input_file = _write_json(tmp_path / "input.json", valid_input)

    with pytest.raises(ValueError, match="Grid physical maximum boundaries must be strictly greater than minimum boundaries"):
        load_and_validate_inputs(input_file, config_file)


def test_mask_length_discretization_mismatch(tmp_path):
    """Verifies ValueError when spatial mask length does not equal expected nx * ny * nz."""
    valid_input, valid_config = _create_valid_schemas()
    config_file = _write_json(tmp_path / "config.json", valid_config)

    # Expected mask len = 2 * 2 * 2 = 8, but we provide 4 elements:
    valid_input["mask"] = [1, 1, 1, 1]
    input_file = _write_json(tmp_path / "input.json", valid_input)

    with pytest.raises(ValueError, match="Mask length mismatch: expected 8 elements, got 4"):
        load_and_validate_inputs(input_file, config_file)


# ============================================================================
# NARRATIVE SECTION 5: Solver Iteration and Numerical Convergence Bounds
# ============================================================================
# Pressure Poisson solver iterations and convergence criteria enforce:
#     max_poisson_iterations > 0
#     poisson_tolerance > 0.0
# ============================================================================


def test_config_poisson_solver_parameters_invalid(tmp_path):
    """Verifies ValueError when Poisson solver iteration limit or tolerance bounds are violated."""
    valid_input, valid_config = _create_valid_schemas()
    input_file = _write_json(tmp_path / "input.json", valid_input)

    # Case A: Non-positive max_poisson_iterations (<= 0)
    valid_config["max_poisson_iterations"] = 0
    config_file_a = _write_json(tmp_path / "config_a.json", valid_config)

    with pytest.raises(ValueError, match="Config schema validation failed"):
        load_and_validate_inputs(input_file, config_file_a)

    # Case B: Non-positive poisson_tolerance (<= 0.0)
    valid_config["max_poisson_iterations"] = 50
    valid_config["poisson_tolerance"] = 0.0
    config_file_b = _write_json(tmp_path / "config_b.json", valid_config)

    with pytest.raises(ValueError, match="Config schema validation failed"):
        load_and_validate_inputs(input_file, config_file_b)