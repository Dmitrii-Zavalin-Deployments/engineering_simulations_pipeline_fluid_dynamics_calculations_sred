/**
 * @file test_orchestrator.cpp
 * @brief Literate Verification Suite for Navier-Stokes Time-Stepping Orchestrator (`orchestrator.cpp`)
 * 
 * @details
 * - What: Validates the Navier-Stokes orchestrator constructor contract guards (such as zero total cells exception handling) 
 *         and full temporal time-stepping integration cycles.
 * - Why: Ensures invalid grid configurations are safely intercepted with `std::invalid_argument` 
 *        and that the multi-phase solver pipeline executes successfully.
 * - How: Exercises constructor failure modes with zero-volume grid dimensions and runs controlled 
 *        simulation steps under a Google Test fixture harness.
 */

#include <gtest/gtest.h>
#include "orchestrator.hpp"
#include <vector>
#include <stdexcept>

// ============================================================================
// SECTION 1 — Constructor Contract & Zero-Cell Exception Validation
// ============================================================================
// Mathematical Rationale:
//   - Total spatial volume is defined as $V = n_x \times n_y \times n_z$. 
//     If any dimension results in zero total cells, memory allocations for trial velocities 
//     and right-hand side buffers become invalid. The orchestrator must throw `std::invalid_argument`.
TEST(OrchestratorTest, ZeroTotalCellsThrowsException) {
    // We configure grid dimensions where nx = 0, resulting in zero total cells.
    navier_stokes_solver::GridDimensions invalid_dims{0, 10, 10, 1.0, 1.0, 1.0};
    navier_stokes_solver::SolverConfig config;

    // Instantiating the orchestrator with zero total cells must trigger the contract guard (line 63).
    EXPECT_THROW(
        navier_stokes_solver::NavierStokesOrchestrator orchestrator(invalid_dims, config),
        std::invalid_argument
    );
}

// ============================================================================
// SECTION 2 — Full Orchestrator Time-Stepping Pipeline Integration
// ============================================================================
// Mathematical Rationale:
//   - Verifies that a valid grid configuration successfully initializes, performs 
//     pre-step ghost syncs, predictor steps, Rhie-Chow face interpolations, Poisson solves, 
//     and corrector velocity projections without exceptions.
TEST(OrchestratorTest, SuccessfulStepExecution) {
    int nx = 3, ny = 3, nz = 3;
    size_t total_cells = static_cast<size_t>(nx * ny * nz);

    navier_stokes_solver::GridDimensions dims{nx, ny, nz, 1.0, 1.0, 1.0};
    navier_stokes_solver::SolverConfig config;
    config.density = 1000.0;
    config.max_poisson_iterations = 10;
    config.poisson_tolerance = 1e-6;

    navier_stokes_solver::NavierStokesOrchestrator orchestrator(dims, config);

    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);
    std::vector<double> gravity = {0.0, 0.0, -9.81};
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);
    std::vector<navier_stokes_solver::BoundaryCondition> bc_list;

    double dt = 0.01;
    double mu = 0.001;

    // Execution of a complete time step must complete successfully across all sub-modules.
    EXPECT_NO_THROW(
        orchestrator.step(
            dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p
        )
    );
}
