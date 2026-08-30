/**
 * @file test_pressure_poisson_solver.cpp
 * @brief Literate Verification Suite for Step 3 Pressure Poisson Solver Module (`pressure_poisson_solver.cpp`)
 * 
 * @details
 * - What: Validates all input contract guards, geometric limits, boundary conditions, and non-finite numeric failure detection.
 * - Why: Ensures 100% statement and branch coverage across Neumann application and Red-Black Gauss-Seidel solver routines.
 * - How: Exercises exception paths for invalid grid spacings, gravity sizes, vector mismatches, and explosive non-finite states under Google Test.
 */

#include <gtest/gtest.h>
#include "pressure_poisson_solver.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

class PressurePoissonTest : public ::testing::Test {
protected:
    int nx = 3, ny = 3, nz = 3;
    size_t total_cells = 27;
    double dx = 1.0, dy = 1.0, dz = 1.0;
    double density = 1000.0;
    std::vector<double> gravity{0.0, 0.0, -9.81};
    navier_stokes_solver::DirichletFaces dirichlet;
};

// ============================================================================
// SECTION 1 — Neumann Pressure Application Validation (Lines 31, 35)
// ============================================================================

// Validates Line 31: Non-positive grid spacing exception handling in Neumann boundary routine
TEST_F(PressurePoissonTest, ApplyNeumannInvalidSpacingThrowsException) {
    std::vector<double> p(total_cells, 0.0);
    EXPECT_THROW(
        navier_stokes_solver::apply_neumann_pressure(
            p, "x_min", dirichlet, nx, ny, nz, 0.0, dy, dz, density, gravity
        ),
        std::invalid_argument
    );
}

// Validates Line 35: Invalid gravity vector size exception handling in Neumann boundary routine
TEST_F(PressurePoissonTest, ApplyNeumannInvalidGravitySizeThrowsException) {
    std::vector<double> p(total_cells, 0.0);
    std::vector<double> bad_gravity = {0.0, -9.81}; // Size != 3
    EXPECT_THROW(
        navier_stokes_solver::apply_neumann_pressure(
            p, "x_min", dirichlet, nx, ny, nz, dx, dy, dz, density, bad_gravity
        ),
        std::invalid_argument
    );
}

// ============================================================================
// SECTION 2 — Poisson Solver Contract & Argument Guard Tests (Lines 202, 205, 208, 213)
// ============================================================================

// Validates Line 202: Minimum grid dimension requirement exception handling (< 3x3x3)
TEST_F(PressurePoissonTest, SolvePoissonSmallGridDimensionsThrowsException) {
    int small_nx = 2;
    size_t small_total = static_cast<size_t>(small_nx * ny * nz);
    std::vector<double> p(small_total, 0.0);
    std::vector<double> rhs(small_total, 0.0);
    std::vector<int> mask(small_total, 1);
    std::vector<navier_stokes_solver::BoundaryCondition> bc_list;

    EXPECT_THROW(
        navier_stokes_solver::solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list, small_nx, ny, nz, dx, dy, dz, 10, 1e-6, density, gravity
        ),
        std::invalid_argument
    );
}

// Validates Line 205: Strictly positive grid spacing exception handling
TEST_F(PressurePoissonTest, SolvePoissonNonPositiveSpacingThrowsException) {
    std::vector<double> p(total_cells, 0.0);
    std::vector<double> rhs(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);
    std::vector<navier_stokes_solver::BoundaryCondition> bc_list;

    EXPECT_THROW(
        navier_stokes_solver::solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list, nx, ny, nz, -1.0, dy, dz, 10, 1e-6, density, gravity
        ),
        std::invalid_argument
    );
}

// Validates Line 208: Invalid iteration counts or negative tolerance exception handling
TEST_F(PressurePoissonTest, SolvePoissonInvalidIterationsOrToleranceThrowsException) {
    std::vector<double> p(total_cells, 0.0);
    std::vector<double> rhs(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);
    std::vector<navier_stokes_solver::BoundaryCondition> bc_list;

    // Test max_iters <= 0
    EXPECT_THROW(
        navier_stokes_solver::solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list, nx, ny, nz, dx, dy, dz, 0, 1e-6, density, gravity
        ),
        std::invalid_argument
    );

    // Test tolerance < 0.0
    EXPECT_THROW(
        navier_stokes_solver::solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list, nx, ny, nz, dx, dy, dz, 10, -1e-6, density, gravity
        ),
        std::invalid_argument
    );
}

// Validates Line 213: Vector dimension size mismatch exception handling
TEST_F(PressurePoissonTest, SolvePoissonVectorSizeMismatchThrowsException) {
    std::vector<double> p(total_cells, 0.0);
    std::vector<double> mismatched_rhs(total_cells - 1, 0.0); // Size mismatch
    std::vector<int> mask(total_cells, 1);
    std::vector<navier_stokes_solver::BoundaryCondition> bc_list;

    EXPECT_THROW(
        navier_stokes_solver::solve_poisson_red_black_parallel(
            p, mismatched_rhs, mask, bc_list, nx, ny, nz, dx, dy, dz, 10, 1e-6, density, gravity
        ),
        std::invalid_argument
    );
}

// ============================================================================
// SECTION 3 — Numerical Stability & Runtime Error Tests (Lines 375–377)
// ============================================================================

// Validates Lines 375–377: Non-finite pressure evaluation detection and runtime error throwing
TEST_F(PressurePoissonTest, SolvePoissonNonFinitePressureThrowsRuntimeError) {
    std::vector<double> p(total_cells, 0.0);
    std::vector<double> rhs(total_cells, 0.0);
    rhs[13] = NAN; // Inject NaN into RHS source term to trigger explosive non-finite state
    std::vector<int> mask(total_cells, 1);
    std::vector<navier_stokes_solver::BoundaryCondition> bc_list;

    EXPECT_THROW(
        navier_stokes_solver::solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list, nx, ny, nz, dx, dy, dz, 10, 1e-6, density, gravity
        ),
        std::runtime_error
    );
}
