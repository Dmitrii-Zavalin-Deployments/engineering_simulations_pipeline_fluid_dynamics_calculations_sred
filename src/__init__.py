"""
Navier-Stokes Solver Package.
Provides Control Plane orchestration, schema ingestion, sovereign state management,
and C++ execution bridge interfaces.
"""

from src.ingestion import load_and_validate_inputs
from src.state import SolverState
from src.cpp_gate import step_simulation
from src.generate_previews import generate_snapshot_preview
from src.archivist import archive_simulation_results

__all__ = [
    "load_and_validate_inputs",
    "SolverState",
    "step_simulation",
    "generate_snapshot_preview",
    "archive_simulation_results",
]
