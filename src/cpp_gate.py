"""
src/cpp_gate.py
C++ Interaction Wrapper Module.
Bridges Python SolverState with the compiled C++ Navier-Stokes engine via Pybind11,
passing the sovereign container object directly to eliminate maintenance drift.
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
    Singleton initializer for the underlying C++ NavierStokesSolver engine.
    Passes the complete sovereign SolverState container object directly to the C++ constructor,
    future-proofing the bridge against field additions.
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided (no defaults allowed).")

    global _cpp_solver_instance
    if _cpp_solver_instance is None:
        logger.info("Initializing C++ NavierStokesSolver engine instance with sovereign SolverState container...")

        # Pass the complete state container object directly to the C++ binding
        _cpp_solver_instance = navier_stokes_cpp.NavierStokesSolver(state)

    return _cpp_solver_instance


def step_simulation(state: SolverState) -> None:
    """
    Executes a single time-integration step through the C++ bridge interface,
    utilizing the sovereign container for array synchronization and parameter access.

    Args:
        state: Sovereign SolverState instance holding physical arrays and simulation parameters.
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided (no defaults allowed).")

    solver = _get_or_create_cpp_solver(state)

    # 1. Prepare C-contiguous array views for zero-copy C++ execution
    fields = np.ascontiguousarray(state.fields, dtype=np.float64)
    mask = np.ascontiguousarray(state.mask, dtype=np.int32)

    # 2. Extract force vectors and synthesize 3D force tensors
    ext_forces = getattr(state, "external_forces", state.input_data.get("external_forces", {}))
    f_vec = ext_forces.get("force_vector", [0.0, 0.0, 0.0])
    g_vec = ext_forces.get("gravity_vector", [0.0, 0.0, 0.0])

    nx, ny, nz = state.nx, state.ny, state.nz
    fx = np.full((nx, ny, nz), float(f_vec[0] + g_vec[0]), dtype=np.float64)
    fy = np.full((nx, ny, nz), float(f_vec[1] + g_vec[1], dtype=np.float64)
    fz = np.full((nx, ny, nz), float(f_vec[2] + g_vec[2], dtype=np.float64)

    # 3. Map boundary conditions from sovereign container to C++ BoundaryCondition structures
    bc_list = []
    bcs = getattr(state, "boundary_conditions", state.input_data.get("boundary_conditions", []))
    for bc_data in bcs:
        bc = navier_stokes_cpp.BoundaryCondition()
        bc.location = bc_data.get("location", "wall")
        bc.type = bc_data.get("type", "no-slip")

        if "values" not in bc_data:
            raise KeyError(f"Missing mandatory 'values' key in boundary condition data: {bc_data}")
        vals = bc_data["values"]

        for key in ["u", "v", "w"]:
            if key not in vals:
                raise KeyError(f"Missing mandatory boundary condition value '{key}' in boundary condition data: {bc_data}")

        bc.u_val = float(vals["u"])
        bc.v_val = float(vals["v"])
        bc.w_val = float(vals["w"])
        bc.scalar_p = float(vals.get("p", 0.0))
        bc_list.append(bc)

    dt = float(getattr(state, "dt", state.input_data["simulation_parameters"]["time_step"]))
    fluid_props = getattr(state, "fluid_properties", state.input_data.get("fluid_properties", {}))
    mu = float(fluid_props["viscosity"])

    try:
        # 4. Execute C++ core time integration step
        solver.step(fields, mask, fx, fy, fz, bc_list, dt, mu)

        # 5. Synchronize updated array buffers back to the sovereign state container
        if not np.shares_memory(state.fields, fields):
            np.copyto(state.fields, fields)

        # 6. Update sovereign state tracking metrics
        state.current_iteration += 1
        state.current_time += dt

    except Exception as e:
        logger.error(f"C++ step execution failed at iteration {getattr(state, 'current_iteration', 0)}: {e}")
        raise RuntimeError(f"C++ execution failure during solver step: {e}") from e
