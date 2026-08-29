/**
 * @file test_corrector.cpp
 * @brief Literate Verification Suite for Step 4 Corrector Velocity Projection Module (`corrector.cpp`)
 * 
 * @details
 * - What: Validates the mathematical projection of trial velocities ($u^*, v^*, w^*$) 
 *         onto the divergence-free subspace using robust mask-aware pressure gradients.
 * - Why: Ensures dimensional safety, contract enforcement, physical parameter validity, 
 *        and numerical stability against non-finite floating-point explosions.
 * - How: Executes boundary condition sweeps, invalid parameter checks, explicit stencil 
 *        evaluations, and forensic error handling under controlled test harnesses.
 */

#include <gtest/gtest.h>
#include "corrector.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

// ============================================================================
// SECTION 1 — Grid Dimension & Geometry Validation Tests
// ============================================================================
// Mathematical Rationale:
//   - The corrector projection requires a minimum interior stencil domain of 3x3x3 
//     to properly compute central differences and boundary-adjacent one-sided gradients.
//   - If grid dimensions fall below this threshold, the solver must throw an explicit exception.
TEST(CorrectorTest, InvalidGridDimensions) {
    // We initialize standard-sized memory buffers for grid validation testing.
    std::vector<double> u(27, 0.0), v(27, 0.0), w(27, 0.0);
    std::vector<double> u_star(27, 0.0), v_star(27, 0.0), w_star(27, 0.0);
    std::vector<double> p(27, 0.0);
    std::vector<int> mask(27, 1);

    // For an invalid grid where nx = 2 (< 3), the dimension contract check:
    //     if (nx < 3 || ny < 3 || nz < 3) throw std::invalid_argument
    EXPECT_THROW(
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            2, 3, 3, 1.0, 1.0, 1.0, 0.01, 1000.0
        ),
        std::invalid_argument
    );
}

// ============================================================================
// SECTION 2 — Grid Spacing & Physical Parameter Positivity Checks
// ============================================================================
// Mathematical Rationale:
//   - Spatial resolution (dx, dy, dz), time step (dt), and fluid density (rho) 
//     serve as direct denominators in gradient operators and projection coefficients 
//     (coeff = dt / rho). They must be strictly positive.
TEST(CorrectorTest, InvalidGridSpacing) {
    std::vector<double> u(27, 0.0), v(27, 0.0), w(27, 0.0);
    std::vector<double> u_star(27, 0.0), v_star(27, 0.0), w_star(27, 0.0);
    std::vector<double> p(27, 0.0);
    std::vector<int> mask(27, 1);

    // When grid spacing dx <= 0.0, spatial inverse operators become undefined.
    EXPECT_THROW(
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            3, 3, 3, 0.0, 1.0, 1.0, 0.01, 1000.0
        ),
        std::invalid_argument
    );
}

TEST(CorrectorTest, InvalidPhysicsParameters) {
    std::vector<double> u(27, 0.0), v(27, 0.0), w(27, 0.0);
    std::vector<double> u_star(27, 0.0), v_star(27, 0.0), w_star(27, 0.0);
    std::vector<double> p(27, 0.0);
    std::vector<int> mask(27, 1);

    // Testing time step dt <= 0.0 constraint violation:
    EXPECT_THROW(
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            3, 3, 3, 1.0, 1.0, 1.0, 0.0, 1000.0
        ),
        std::invalid_argument
    );

    // Testing density rho <= 0.0 constraint violation:
    EXPECT_THROW(
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            3, 3, 3, 1.0, 1.0, 1.0, 0.01, 0.0
        ),
        std::invalid_argument
    );
}

// ============================================================================
// SECTION 3 — Memory Allocation Contract Verification
// ============================================================================
// Mathematical Rationale:
//   - All vector buffers must match the total spatial volume (nx * ny * nz). 
//     Mismatched sizes trigger an immediate contract violation exception.
TEST(CorrectorTest, VectorSizeMismatch) {
    // We intentionally create a buffer size of 10 while grid dimensions specify 27 cells.
    std::vector<double> u(10, 0.0); 
    std::vector<double> v(27, 0.0), w(27, 0.0);
    std::vector<double> u_star(27, 0.0), v_star(27, 0.0), w_star(27, 0.0);
    std::vector<double> p(27, 0.0);
    std::vector<int> mask(27, 1);

    EXPECT_THROW(
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            3, 3, 3, 1.0, 1.0, 1.0, 0.01, 1000.0
        ),
        std::invalid_argument
    );
}

// ============================================================================
// SECTION 4 — Forensic Numerical Audit & Non-Finite Exception Handling
// ============================================================================
// Mathematical Rationale:
//   - If floating-point anomalies (NaN or Infinity) propagate into trial velocity fields, 
//     the forensic auditor detects the failure, logs coordinates, and aborts execution.
TEST(CorrectorTest, NonFiniteVelocityExplosion) {
    std::vector<double> u(27, 0.0), v(27, 0.0), w(27, 0.0);
    std::vector<double> u_star(27, 0.0), v_star(27, 0.0), w_star(27, 0.0);
    std::vector<double> p(27, 0.0);
    std::vector<int> mask(27, 1);

    // We inject a NaN value into the center cell of the trial velocity field.
    u_star[13] = NAN;

    // The solver must intercept this during the projection pass and throw a runtime_error.
    EXPECT_THROW(
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            3, 3, 3, 1.0, 1.0, 1.0, 0.01, 1000.0
        ),
        std::runtime_error
    );
}

// ============================================================================
// SECTION 5 — Boundary Stencil & Mask-Aware Gradient Verification
// ============================================================================
// Mathematical Rationale:
//   - Non-fluid cells (mask != 1, representing solid boundaries `0` or walls `-1`) 
//     must execute one-sided gradient branches and get clamped to zero velocity.
TEST(CorrectorTest, BoundaryAndOneSidedGradients) {
    int nx = 3, ny = 3, nz = 3;
    size_t total_cells = static_cast<size_t>(nx * ny * nz);

    std::vector<double> u(total_cells, 0.0), v(total_cells, 0.0), w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 1.0), v_star(total_cells, 1.0), w_star(total_cells, 1.0);
    std::vector<double> p(total_cells, 10.0);
    std::vector<int> mask(total_cells, 1);

    // We configure diverse boundary masks to exercise forward, backward, and solid clamping branches:
    mask[0] = 0;   // Solid boundary cell
    mask[2] = -1;  // Wall-adjacent boundary cell

    // Execution must complete successfully across all stencil branches without throwing.
    EXPECT_NO_THROW(
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, 1.0, 1.0, 1.0, 0.01, 1000.0
        )
    );

    // Non-fluid cells must be strictly clamped to zero velocity:
    //     u[idx] = 0.0, v[idx] = 0.0, w[idx] = 0.0
    EXPECT_EQ(u[0], 0.0);
    EXPECT_EQ(v[0], 0.0);
    EXPECT_EQ(w[0], 0.0);
}
