"""
src/state.py
Sovereign Container Module.
Maintains the complete operational state of the simulation, holding input metadata,
dynamic 4D field buffers (u, v, w, p), and physical boundary enforcement.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("Solver.State")


class SolverState:
    """
    Sovereign state container managing field arrays, grid parameters, 
    and constraint evaluations across simulation iterations.
    """

    def __init__(self, input_data: dict[str, Any], config_data: dict[str, Any]):
        if input_data is None:
            raise ValueError("FATAL ERROR: input_data must be explicitly provided (no defaults allowed).")
        if config_data is None:
            raise ValueError("FATAL ERROR: config_data must be explicitly provided (no defaults allowed).")

        self.input_data: dict[str, Any] = input_data
        self.config: dict[str, Any] = config_data

        # Grid parameters
        if "grid" not in input_data or input_data["grid"] is None:
            raise KeyError("Non-default policy violation: missing required 'grid' section in input_data.")
        grid = input_data["grid"]
        
        for k in ["nx", "ny", "nz", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]:
            if k not in grid or grid[k] is None:
                raise KeyError(f"Non-default policy violation in 'grid': missing required key '{k}'.")

        self.nx: int = int(grid["nx"])
        self.ny: int = int(grid["ny"])
        self.nz: int = int(grid["nz"])

        self.x_min: float = float(grid["x_min"])
        self.x_max: float = float(grid["x_max"])
        self.y_min: float = float(grid["y_min"])
        self.y_max: float = float(grid["y_max"])
        self.z_min: float = float(grid["z_min"])
        self.z_max: float = float(grid["z_max"])

        self.dx: float = (self.x_max - self.x_min) / self.nx
        self.dy: float = (self.y_max - self.y_min) / self.ny
        self.dz: float = (self.z_max - self.z_min) / self.nz

        # Sub-schemas with strict presence checks
        for sec in [
            "fluid_properties",
            "initial_conditions",
            "simulation_parameters",
            "boundary_conditions",
            "external_forces",
            "domain_configuration",
            "physical_constraints",
            "mask",
        ]:
            if sec not in input_data or input_data[sec] is None:
                raise KeyError(f"Non-default policy violation: missing required section '{sec}' in input_data.")

        self.grid: dict[str, Any] = grid
        self.fluid_properties: dict[str, Any] = input_data["fluid_properties"]
        self.initial_conditions: dict[str, Any] = input_data["initial_conditions"]
        self.simulation_parameters: dict[str, Any] = input_data["simulation_parameters"]
        self.boundary_conditions: list[dict[str, Any]] = input_data["boundary_conditions"]
        self.external_forces: dict[str, Any] = input_data["external_forces"]
        self.domain_configuration: dict[str, Any] = input_data["domain_configuration"]
        self.physical_constraints: dict[str, Any] = input_data["physical_constraints"]

        # 4D fields buffer: shape (4, nx, ny, nz) -> [0]: u, [1]: v, [2]: w, [3]: p
        self.fields: np.ndarray = np.zeros((4, self.nx, self.ny, self.nz), dtype=np.float64)
        
        ic_v = self.initial_conditions["velocity"]
        self.fields[0, :, :, :] = ic_v[0]
        self.fields[1, :, :, :] = ic_v[1]
        self.fields[2, :, :, :] = ic_v[2]
        self.fields[3, :, :, :] = self.initial_conditions["pressure"]

        # Mask buffer: shape (nx, ny, nz)
        self.mask: np.ndarray = np.array(input_data["mask"], dtype=np.int32).reshape(
            (self.nx, self.ny, self.nz)
        )

        # Simulation execution tracking
        self.current_iteration: int = 0
        self.current_time: float = 0.0
        self.dt: float = float(self.simulation_parameters["time_step"])
        self.total_time: float = float(self.simulation_parameters["total_time"])
        self.total_iterations: int = round(self.total_time / self.dt)
        self.output_interval: int = int(self.simulation_parameters["output_interval"])

        self.history_logs: list[dict[str, Any]] = []

    def enforce_physical_constraints(self) -> None:
        """
        Validates velocity and pressure fields against physical bounds specified in input schema,
        clamping values to prevent unphysical numerical divergence.
        """
        min_v = float(self.physical_constraints["min_velocity"])
        max_v = float(self.physical_constraints["max_velocity"])
        min_p = float(self.physical_constraints["min_pressure"])
        max_p = float(self.physical_constraints["max_pressure"])

        # Check for potential explosive NaNs/Infs
        if not np.isfinite(self.fields).all():
            logger.critical(f"Non-finite values (NaN/Inf) detected at iteration {self.current_iteration}.")
            raise ArithmeticError("Numerical instability detected: fields contain NaN or Inf values.")

        # Velocity clamping
        u_clamped = np.clip(self.fields[0], min_v, max_v)
        v_clamped = np.clip(self.fields[1], min_v, max_v)
        w_clamped = np.clip(self.fields[2], min_v, max_v)

        # Pressure clamping
        p_clamped = np.clip(self.fields[3], min_p, max_p)

        self.fields[0] = u_clamped
        self.fields[1] = v_clamped
        self.fields[2] = w_clamped
        self.fields[3] = p_clamped
