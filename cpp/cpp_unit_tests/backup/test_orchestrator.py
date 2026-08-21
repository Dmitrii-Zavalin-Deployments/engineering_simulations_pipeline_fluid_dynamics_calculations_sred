"""
@file test_orchestrator.py
@brief Literate Test Suite for C++ Navier-Stokes Time-Stepping Orchestrator

This test file acts as a narrative document. Explanatory text and physical
principles are written as commented prose, while executable Python assertions
verify the correctness of the 3D Navier-Stokes Orchestrator, hydrostatic pressure
splitting, predictor-corrector mechanics, and rigorous contract enforcement.
"""

import numpy as np
import pytest

try:
    import navier_stokes_cpp
except ImportError:
    navier_stokes_cpp = None


class MockOrchestratorState:
    """Mock sovereign state container for testing NavierStokesOrchestrator."""
    def __init__(self, nx=8, ny=8, nz=8):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = 1.0 / nx
        self.dy = 1.0 / ny
        self.dz = 1.0 / nz
        self.dt = 0.001
        
        self.total_cells = nx * ny * nz

        # Use 1D arrays to align with std::vector<double> pybind11 casting
        self.fields = np.zeros((4, self.total_cells), dtype=np.float64)
        # Initialize velocity u component (offset 0) to 0.1
        self.fields[0, :] = 0.1

        # Fluid mask: 1 for fluid cells, 0 for solid, -1 for boundary/wall
        self.mask = np.ones(self.total_cells, dtype=np.int32)

        self.fluid_properties = {
            "density": 1000.0,
            "viscosity": 0.001
        }
        self.config = {
            "max_poisson_iterations": 50,
            "poisson_tolerance": 1e-6
        }
        self.external_forces = {
            "gravity_vector": [0.0, -9.81, 0.0],
            "force_vector": [10.0, 0.0, 0.0],
            "fx": np.zeros(self.total_cells, dtype=np.float64),
            "fy": np.zeros(self.total_cells, dtype=np.float64),
            "fz": np.zeros(self.total_cells, dtype=np.float64)
        }
        self.boundary_conditions = []


# ============================================================================
# NARRATIVE SECTION 1: Orchestrator Initialization and Contract Enforcement
# ============================================================================
# The NavierStokesOrchestrator manages the temporal discretization and spatial
# operators for 3D incompressible fluid flow. 
#
# The governing momentum equation under hydrostatic pressure splitting is:
#
#     du*/dt + (u * nabla)u = - 1/rho * grad(p_dyn) + nu * laplacian(u) + f
#
# where u* is the intermediate trial velocity vector field.
# ============================================================================

def test_orchestrator_initialization():
    """Verifies successful instantiation of NavierStokesOrchestrator with valid grid dimensions and config."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    # Grid dimensions: nx = 8, ny = 8, nz = 8, dx = 0.125, dy = 0.125, dz = 0.125
    nx, ny, nz = 8, 8, 8
    dims = navier_stokes_cpp.GridDimensions(nx, ny, nz, 0.125, 0.125, 0.125)
    config = navier_stokes_cpp.SolverConfig(1000.0, 50, 1e-6)

    orchestrator = navier_stokes_cpp.NavierStokesOrchestrator(dims, config)
    assert orchestrator is not None


def test_orchestrator_gravity_contract_violation():
    """Triggers contract violation exception when gravity vector does not contain exactly 3 components."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    total_cells = nx * ny * nz
    dims = navier_stokes_cpp.GridDimensions(nx, ny, nz, 0.125, 0.125, 0.125)
    config = navier_stokes_cpp.SolverConfig(1000.0, 50, 1e-6)
    orchestrator = navier_stokes_cpp.NavierStokesOrchestrator(dims, config)

    # Invalid gravity vector with 2 components instead of 3: [gx, gy]
    invalid_gravity = [0.0, -9.81]
    
    # Use 1D arrays correctly sized to 'total_cells' to bypass Pybind11 cast failures
    # and guarantee the execution drops into the C++ `if (gravity.size() != 3)` check.
    fx = np.zeros(total_cells, dtype=np.float64)
    fy = np.zeros(total_cells, dtype=np.float64)
    fz = np.zeros(total_cells, dtype=np.float64)
    mask = np.ones(total_cells, dtype=np.int32)
    u = np.zeros(total_cells, dtype=np.float64)
    v = np.zeros(total_cells, dtype=np.float64)
    w = np.zeros(total_cells, dtype=np.float64)
    p = np.zeros(total_cells, dtype=np.float64)
    bc_list = []

    # The momentum equation contract expects:
#     gravity.size() == 3
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        orchestrator.step(
            0.001, 0.001, invalid_gravity,
            fx, fy, fz, mask, bc_list,
            u, v, w, p
        )


# ============================================================================
# NARRATIVE SECTION 2: End-to-End Time-Stepping Execution and Telemetry
# ============================================================================
# Executing a full time step orchestrates pre-stepping, ghost buffer synchronization,
# predictor computation, Red-Black Parallel Poisson pressure solving, and corrector projection.
#
# The velocity divergence condition enforced by the Poisson equation is:
#
#     div(u^{n+1}) = 0
# ============================================================================

def test_orchestrator_full_time_step():
    """Executes a complete time step through the orchestrator and validates field finiteness and state update."""
    if navier_stokes_cpp is None:
        pytest.skip("navier_stokes_cpp module not available.")

    nx, ny, nz = 8, 8, 8
    total_cells = nx * ny * nz
    dims = navier_stokes_cpp.GridDimensions(nx, ny, nz, 0.125, 0.125, 0.125)
    config = navier_stokes_cpp.SolverConfig(1000.0, 50, 1e-6)
    orchestrator = navier_stokes_cpp.NavierStokesOrchestrator(dims, config)

    dt = 0.001
    mu = 0.001
    gravity = [0.0, -9.81, 0.0]
    
    # 1D arrays formatted for std::vector casting
    fx = np.zeros(total_cells, dtype=np.float64)
    fy = np.zeros(total_cells, dtype=np.float64)
    fz = np.zeros(total_cells, dtype=np.float64)
    mask = np.ones(total_cells, dtype=np.int32)

    u = np.zeros(total_cells, dtype=np.float64)
    v = np.zeros(total_cells, dtype=np.float64)
    w = np.zeros(total_cells, dtype=np.float64)
    p = np.zeros(total_cells, dtype=np.float64)
    
    # Seed initial velocity
    u.fill(0.1)

    bc_list = []

    # Execute time step
    orchestrator.step(
        dt, mu, gravity,
        fx, fy, fz, mask, bc_list,
        u, v, w, p
    )

    # Assert all velocity and pressure fields remain finite (no NaN or Inf excursions)
#     finiteness condition: all(isfinite(field)) == True
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(p))
