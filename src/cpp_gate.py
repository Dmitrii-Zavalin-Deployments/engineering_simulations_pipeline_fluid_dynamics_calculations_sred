"""
src/cpp_gate.py
C++ Interaction Wrapper Module.
Bridges Python SolverState with the compiled C++ Navier-Stokes engine via Pybind11,
passing the sovereign container directly to establish zero-copy memory binding 
and eliminate parameter drift.
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
    Passes the complete sovereign SolverState container directly to the C++ constructor,
    binding Python memory directly to the C++ execution engine.
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
    utilizing direct sovereign container reference for in-place RAM mutation.

    Args:
        state: Sovereign SolverState instance holding physical arrays and simulation parameters.
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided (no defaults allowed).")

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
