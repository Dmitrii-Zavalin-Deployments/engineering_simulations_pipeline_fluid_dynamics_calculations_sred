"""
tests/conftest.py
Pytest fixtures providing test data and workspace directories for CLI-driven integration tests,
dynamically syncing configuration data directly from production config/config.json.
"""

import json
import os

import pytest


@pytest.fixture
def valid_input_data():
    """Returns a valid 2x2x2 domain input configuration dictionary for manual inspection."""
    nx, ny, nz = 3, 3, 3
    return {
        "physical_constraints": {
            "min_velocity": -10.0,
            "max_velocity": 10.0,
            "min_pressure": -100.0,
            "max_pressure": 100.0,
        },
        "domain_configuration": {
            "type": "INTERNAL",
            "reference_velocity": [0.0, 0.0, 0.0],
        },
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "z_min": 0.0,
            "z_max": 1.0,
            "nx": nx,
            "ny": ny,
            "nz": nz,
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.01,
        },
        "initial_conditions": {
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0,
        },
        "simulation_parameters": {
            "time_step": 0.001,
            "total_time": 0.003,  # 3 iterations
            "output_interval": 1,
        },
        "boundary_conditions": [
            {
                "location": "wall",
                "type": "no-slip",
                "values": {"u": 0.0, "v": 0.0, "w": 0.0},
            }
        ],
        "mask": [0] * (nx * ny * nz),
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "gravity_vector": [0.0, -9.81, 0.0],
        },
    }


@pytest.fixture
def workspace_folder(tmp_path, valid_input_data):
    """
    Creates an input/output workspace folder containing input JSON files
    and mirrors the production config/config.json to satisfy CLI requirements.
    """
    io_folder = tmp_path / "io_workspace"
    io_folder.mkdir(parents=True, exist_ok=True)

    input_file_name = "navier_stokes_input.json"
    config_file_name = "config.json"

    input_path = io_folder / input_file_name
    config_path = io_folder / config_file_name

    # Write valid input data
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(valid_input_data, f)

    # Dynamically load and mirror the production configuration file
    prod_config_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "config", "config.json"),
        "config/config.json",
    ]

    prod_config = None
    for candidate in prod_config_candidates:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                prod_config = json.load(f)
            break

    if prod_config is None:
        raise FileNotFoundError("Critical: Production config/config.json not found for test workspace synchronization.")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(prod_config, f)

    return {
        "folder": str(io_folder),
        "input_file_name": input_file_name,
        "config_path": str(config_path),
        "input_path": str(input_path),
    }
