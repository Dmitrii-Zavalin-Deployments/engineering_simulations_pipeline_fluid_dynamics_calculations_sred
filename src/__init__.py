"""
Navier-Stokes Solver Package.
Provides Control Plane orchestration, schema ingestion, sovereign state management,
and C++ execution bridge interfaces.
"""

from src.archivist import archive_simulation_results
from src.cpp_gate import step_simulation
from src.generate_previews import generate_snapshot_preview
from src.ingestion import load_and_validate_inputs
from src.state import SolverState

__all__ = [
    "SolverState",
    "archive_simulation_results",
    "generate_snapshot_preview",
    "load_and_validate_inputs",
    "step_simulation",
]
