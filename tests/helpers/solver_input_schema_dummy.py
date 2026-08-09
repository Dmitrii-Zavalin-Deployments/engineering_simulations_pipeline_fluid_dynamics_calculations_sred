# tests/helpers/solver_input_schema_dummy.py

"""
Archivist Testing: Explicit Input Hydration.

Compliance:
- Rule 5: Deterministic Initialization (No default fallbacks).
- Rule 8: API Minimalism (Primary interfaces only).
"""

from typing import Any

from src.common.solver_input import SolverInput


def get_explicit_solver_config(nx: int, ny: int, nz: int) -> dict[str, Any]:
    """
    Returns a strictly typed dictionary for input hydration.
    No hardcoded defaults; all simulation parameters are explicit.
    """
    return {
        "domain_configuration": {
            "type": "INTERNAL",
            "reference_velocity": [0.0, 0.0, 0.0]
        },
        "grid": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": nx, "ny": ny, "nz": nz,
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.001,
        },
        "initial_conditions": {
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0,
        },
        "simulation_parameters": {
            "time_step": 0.01,
            "total_time": 0.1,
            "output_interval": 1,
        },
        # --- NEW PHYSICAL CONSTRAINTS BLOCK ---
        "physical_constraints": {
            "min_velocity": -100.0,
            "max_velocity": 100.0,
            "min_pressure": -1e6,
            "max_pressure": 1e6,
        },
        # --------------------------------------
        "boundary_conditions": [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}},
            {"location": "x_max", "type": "outflow", "values": {"p": 0.0}},
            {"location": "y_min", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0}},
            {"location": "y_max", "type": "free-slip", "values": {"u": 1.0, "w": 0.0}},
            {"location": "z_min", "type": "pressure", "values": {"p": 101325.0}},
            {"location": "z_max", "type": "pressure", "values": {"p": 0.0}},
            {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0}},
        ],
        "mask": [1] * (nx * ny * nz),
        "external_forces": {
            "force_vector": [0.0, 0.0, -9.81]
        },
    }

def create_validated_input(nx: int = 2, ny: int = 2, nz: int = 2) -> SolverInput:
    """
    Hydrates a SolverInput object from explicit parameters.
    Ensures that simulation initialization is reproducible.
    """
    config = get_explicit_solver_config(nx, ny, nz)
    return SolverInput.from_dict(config)