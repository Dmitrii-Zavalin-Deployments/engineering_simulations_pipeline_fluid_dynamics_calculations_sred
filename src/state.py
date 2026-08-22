"""
Core State Container Module.
Maintains the complete operational state of the simulation, holding input metadata,
dynamic 4D field buffers (u, v, w, p), and physical boundary enforcement.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("Solver.State")


class SolverState:
    """
    Simulation state container managing field arrays, grid parameters, 
    and constraint evaluations across simulation iterations under strict non-default policies.
    """

    def __init__(self, input_data: dict[str, Any], config_data: dict[str, Any]):
        if input_data is None:
            raise ValueError("FATAL ERROR: input_data must be explicitly provided (no defaults allowed).")
        if config_data is None:
            raise ValueError("FATAL ERROR: config_data must be explicitly provided (no defaults allowed).")

        self.input_data: dict[str, Any] = input_data
        self.config: dict[str, Any] = config_data

        # Grid parameters presence and value validation
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

        # Sub-schemas with strict presence checks matching the schema
        for sec in [
            "fluid_properties",
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

        self.simulation_parameters: dict[str, Any] = input_data["simulation_parameters"]
        sim_params = self.simulation_parameters
        for k in ["time_step", "total_time", "output_interval"]:
            if k not in sim_params or sim_params[k] is None:
                raise KeyError(f"Non-default policy violation in 'simulation_parameters': missing required key '{k}'.")

        self.boundary_conditions: list[Any] = input_data["boundary_conditions"]
        self.external_forces: dict[str, Any] = input_data["external_forces"]
        self.domain_configuration: dict[str, Any] = input_data["domain_configuration"]
        self.physical_constraints: dict[str, Any] = input_data["physical_constraints"]

        pc = self.physical_constraints
        for k in ["min_velocity", "max_velocity", "min_pressure", "max_pressure"]:
            if k not in pc or pc[k] is None:
                raise KeyError(f"Non-default policy violation in 'physical_constraints': missing required key '{k}'.")

        # 4D fields buffer: shape (4, nx, ny, nz) -> [0]: u, [1]: v, [2]: w, [3]: p
        self.fields: np.ndarray = np.zeros((4, self.nx, self.ny, self.nz), dtype=np.float64, order='C')

        # Convenience slices sharing memory views with self.fields
        self.u = self.fields[0]
        self.v = self.fields[1]
        self.w = self.fields[2]
        self.p = self.fields[3]

        # Mask buffer: shape (nx, ny, nz)
        self.mask: np.ndarray = np.array(input_data["mask"], dtype=np.int32, order='C').reshape(
            (self.nx, self.ny, self.nz)
        )

        # Simulation execution tracking
        self.current_iteration: int = 0
        self.current_time: float = 0.0
        self.dt: float = float(sim_params["time_step"])
        self.total_time: float = float(sim_params["total_time"])
        self.total_iterations: int = round(self.total_time / self.dt)
        self.output_interval: int = int(sim_params["output_interval"])

        self.history_logs: list[dict[str, Any]] = []
        self._cpp_solver = None

        logger.info(f"Initialized SolverState: Grid {self.nx}x{self.ny}x{self.nz}, Iterations: {self.total_iterations}")

    def enforce_physical_constraints(self) -> None:
        """
        Validates fields for numerical stability (NaN/Inf detection),
        allowing unconstrained physical evolution.
        """
        if not np.isfinite(self.fields).all():
            logger.critical(f"Non-finite values (NaN/Inf) detected at iteration {self.current_iteration}.")
            raise ArithmeticError("Numerical instability detected: fields contain NaN or Inf values.")

    def get_boundary_condition_dicts(self) -> list[dict[str, Any]]:
        """
        Safely retrieves boundary conditions as dictionaries, 
        handling both raw dicts and converted C++ BoundaryCondition objects.
        """
        bc_list = []
        for bc in self.boundary_conditions:
            if isinstance(bc, dict):
                bc_list.append(bc)
            else:
                bc_list.append({
                    "location": getattr(bc, "location", ""),
                    "type": getattr(bc, "type", ""),
                    "values": {
                        "u": getattr(bc, "u_val", 0.0),
                        "v": getattr(bc, "v_val", 0.0),
                        "w": getattr(bc, "w_val", 0.0),
                        "p": getattr(bc, "scalar_p", 0.0)
                    }
                })
        return bc_list
