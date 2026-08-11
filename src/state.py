"""
src/state.py
Sovereign Container Module.
Maintains the complete operational state of the simulation, holding input metadata,
dynamic 4D field buffers (u, v, w, p), snapshot logs, and physical boundary enforcement.
"""

import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger("Solver.State")


class SolverState:
    """
    Sovereign state container managing field arrays, grid parameters, 
    and constraint evaluations across simulation iterations.
    """

    def __init__(self, input_data: Dict[str, Any], config_data: Dict[str, Any]):
        self.input_data: Dict[str, Any] = input_data
        self.config: Dict[str, Any] = config_data

        # Grid parameters
        grid = input_data["grid"]
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

        # Sub-schemas
        self.grid: Dict[str, Any] = grid
        self.fluid_properties: Dict[str, Any] = input_data["fluid_properties"]
        self.initial_conditions: Dict[str, Any] = input_data["initial_conditions"]
        self.simulation_parameters: Dict[str, Any] = input_data["simulation_parameters"]
        self.boundary_conditions: List[Dict[str, Any]] = input_data["boundary_conditions"]
        self.external_forces: Dict[str, Any] = input_data["external_forces"]
        self.domain_configuration: Dict[str, Any] = input_data["domain_configuration"]
        self.physical_constraints: Dict[str, Any] = input_data["physical_constraints"]

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
        self.total_iterations: int = int(round(self.total_time / self.dt))
        self.output_interval: int = int(self.simulation_parameters["output_interval"])

        self.history_logs: List[Dict[str, Any]] = []
        self.snapshot_records: List[Dict[str, Any]] = []

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

    def record_snapshot(self, preview_file_path: str = None) -> None:
        """
        Appends iteration summary statistics and associated preview metadata to snapshot records.
        """
        mag_v = np.sqrt(self.fields[0] ** 2 + self.fields[1] ** 2 + self.fields[2] ** 2)
        snapshot = {
            "iteration": self.current_iteration,
            "time": self.current_time,
            "min_velocity": float(np.min(mag_v)),
            "max_velocity": float(np.max(mag_v)),
            "mean_velocity": float(np.mean(mag_v)),
            "min_pressure": float(np.min(self.fields[3])),
            "max_pressure": float(np.max(self.fields[3])),
            "mean_pressure": float(np.mean(self.fields[3])),
            "preview_file": preview_file_path,
        }
        self.snapshot_records.append(snapshot)
        logger.debug(f"Recorded state snapshot at iteration {self.current_iteration} (t={self.current_time:.4f}s)")
