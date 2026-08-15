"""
src/cpp_gate.py
C++ Interaction Wrapper Module.
Bridges Python SolverState with the compiled C++ Navier-Stokes engine via Pybind11,
passing the sovereign container directly to establish zero-copy memory binding 
and eliminate parameter drift.
"""

import logging
from typing import Any

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


def _dict_to_boundary_condition(bc_dict: dict) -> Any:
    """Instantiates and populates a C++ BoundaryCondition object from a Python dict, mapping nested values to C++ fields."""
    bc_obj = navier_stokes_cpp.BoundaryCondition()
    for key, value in bc_dict.items():
        if key == "values" and isinstance(value, dict):
            if "u" in value and hasattr(bc_obj, "u_val"):
                bc_obj.u_val = float(value["u"])
            if "v" in value and hasattr(bc_obj, "v_val"):
                bc_obj.v_val = float(value["v"])
            if "w" in value and hasattr(bc_obj, "w_val"):
                bc_obj.w_val = float(value["w"])
            if "p" in value and hasattr(bc_obj, "scalar_p"):
                bc_obj.scalar_p = float(value["p"])
        elif hasattr(bc_obj, key):
            setattr(bc_obj, key, value)
    return bc_obj


def _convert_boundary_conditions(state: SolverState) -> None:
    """Converts dictionary boundary conditions to C++ BoundaryCondition objects in-place."""
    if hasattr(state, "boundary_conditions") and state.boundary_conditions:
        state.boundary_conditions = [
            _dict_to_boundary_condition(bc) if isinstance(bc, dict) else bc
            for bc in state.boundary_conditions
        ]


def _get_or_create_cpp_solver(state: SolverState) -> Any:
    """
    Singleton initializer for the underlying C++ NavierStokesSolver engine.
    Passes the complete sovereign SolverState container directly to the C++ constructor,
    binding Python memory directly to the C++ execution engine.
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided (no defaults allowed).")

    global _cpp_solver_instance
    if _cpp_solver_instance is None:
        _convert_boundary_conditions(state)
        logger.info("Initializing C++ NavierStokesSolver engine instance with sovereign SolverState container...")
        _cpp_solver_instance = navier_stokes_cpp.NavierStokesSolver(state)

    return _cpp_solver_instance


def step_simulation(state: SolverState) -> None:
    """
    Executes a single time-integration step through the C++ bridge interface,
    utilizing direct sovereign container reference for in-place RAM mutation.

    Args:
        state: Sovereign SolverState instance holding physical arrays and simulation parameters.
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided (no defaults allowed).")

    _convert_boundary_conditions(state)
    solver = _get_or_create_cpp_solver(state)

    try:
        # Execute C++ core time integration step.
        # C++ reads and writes directly to state.fields in-place via zero-copy memory binding.
        solver.step(state)

        # Update sovereign state tracking metrics upon successful step completion
        dt = float(getattr(state, "dt", state.input_data["simulation_parameters"]["time_step"]))
        state.current_iteration += 1
        state.current_time += dt

    except Exception as e:
        logger.error(f"C++ step execution failed at iteration {getattr(state, 'current_iteration', 0)}: {e}")
        raise RuntimeError(f"C++ execution failure during solver step: {e}") from e
