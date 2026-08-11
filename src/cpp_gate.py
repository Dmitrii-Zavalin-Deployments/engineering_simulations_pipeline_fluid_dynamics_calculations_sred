"""
src/cpp_gate.py
C++ Interaction Wrapper Module.
Bridges Python SolverState with the compiled C++ Navier-Stokes engine via Pybind11,
handling zero-copy array passes and boundary condition mapping.
"""

import logging
import numpy as np
from src.state import SolverState

try:
    import navier_stokes_cpp
except ImportError as e:
    raise ImportError(
        "Failed to import compiled C++ module 'navier_stokes_cpp'. "
        "Ensure the C++ library is built and Python path includes the build directory."
    ) from e

logger = logging.getLogger("Solver.CppGate")

_cpp_solver_instance = None


def _get_or_create_cpp_solver(state: SolverState) -> navier_stokes_cpp.NavierStokesSolver:
    """
    Singleton initializer for the underlying C++ NavierStokesSolver.
    Extracts configuration parameters and constructs the C++ solver object.
    """
    global _cpp_solver_instance
    if _cpp_solver_instance is None:
        logger.info("Initializing C++ NavierStokesSolver engine instance...")

        _cpp_solver_instance = navier_stokes_cpp.NavierStokesSolver(
            nx=state.nx,
            ny=state.ny,
            nz=state.nz,
            dx=state.dx,
            dy=state.dy,
            dz=state.dz,
            max_poisson_iters=int(state.config["max_poisson_iterations"]),
            poisson_tolerance=float(state.config["poisson_tolerance"]),
            density=float(state.fluid_properties["density"]),
        )
    return _cpp_solver_instance


def step_simulation(state: SolverState) -> None:
    """
    Executes a single time-integration step through the C++ bridge interface.
    
    Args:
        state: SolverState instance holding physical arrays and simulation parameters.
    """
    solver = _get_or_create_cpp_solver(state)

    # Ensure memory layout matches expected contiguous C-order
    fields = np.ascontiguousarray(state.fields, dtype=np.float64)
    mask = np.ascontiguousarray(state.mask, dtype=np.int32)

    # Construct force fields (fx, fy, fz) by combining uniform force and gravity vectors
    f_vec = state.external_forces["force_vector"]
    g_vec = state.external_forces["gravity_vector"]

    fx = np.full((state.nx, state.ny, state.nz), f_vec[0] + g_vec[0], dtype=np.float64)
    fy = np.full((state.nx, state.ny, state.nz), f_vec[1] + g_vec[1], dtype=np.float64)
    fz = np.full((state.nx, state.ny, state.nz), f_vec[2] + g_vec[2], dtype=np.float64)

    # Map Python boundary condition list to C++ BoundaryCondition structures
    bc_list = []
    for bc_data in state.boundary_conditions:
        bc = navier_stokes_cpp.BoundaryCondition()
        bc.location = bc_data["location"]
        bc.type = bc_data["type"]

        vals = bc_data.get("values", {})
        bc.u_val = float(vals.get("u", 0.0))
        bc.v_val = float(vals.get("v", 0.0))
        bc.w_val = float(vals.get("w", 0.0))
        bc.scalar_p = float(vals.get("p", 0.0))
        bc_list.append(bc)

    dt = float(state.dt)
    mu = float(state.fluid_properties["viscosity"])

    try:
        # Pass mutable fields and immutable buffers to C++ step kernel
        solver.step(fields, mask, fx, fy, fz, bc_list, dt, mu)

        # Update time and step counters upon successful execution
        state.current_iteration += 1
        state.current_time += dt

    except Exception as e:
        logger.error(f"C++ step execution failed at iteration {state.current_iteration}: {e}")
        raise RuntimeError(f"C++ execution failure during solver step: {e}") from e
