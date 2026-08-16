/**
 * @file test_pressure_poisson_solver.cpp
 * @brief Literate Test Suite for Pressure Poisson Solver and Boundary Conditions
 *
 * This test file acts as a narrative document. Explanatory text and physical 
 * equations are written as commented prose using ASCII formatting, while the executable C++ assertions 
 * verify Neumann boundary applications, solid cell pressure averaging, error-handling paths,
 * and Red-Black Gauss-Seidel Poisson solver convergence.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <string>
#include "pressure_poisson_solver.hpp"
#include "orchestrator.hpp"
#include "grid_math.hpp"

using namespace navier_stokes_solver;

// ============================================================================
// NARRATIVE SECTION 1: Neumann Pressure Boundary Conditions
// ============================================================================
// Neumann boundary conditions impose a zero-gradient constraint normal to the 
// boundary surfaces (dp/dn = 0). Numerically, this sets the boundary cell pressure 
// equal to the adjacent interior cell pressure adjusted for hydrostatic body forces:
//        p_boundary = p_interior - (density * g_n * d_n)
// Here we verify all directional variants ("x_min", "x_max", "y_min", "y_max", 
// "z_min", "z_max") as well as the composite "wall" boundary type, and defensive 
// exception handling for grid spacing and gravity vector dimensions.
// ============================================================================

TEST(PressurePoissonTest, ApplyNeumannPressureDirections) {
    int nx = 5, ny = 5, nz = 5;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // Test x_min boundary
    {
        std::vector<double> p(total_cells, 0.0);
        p[1 + nx * (2 + ny * 2)] = 10.5; // Interior neighbor at i=1
        apply_neumann_pressure(p, "x_min", nx, ny, nz, 1.0, 1.0, 1.0, 1.0, {0.0, 0.0, 0.0});
        int idx_boundary = 0 + nx * (2 + ny * 2);
        EXPECT_NEAR(p[idx_boundary], 10.5, 1e-12);
    }

    // Test x_max boundary
    {
        std::vector<double> p(total_cells, 0.0);
        p[(nx - 2) + nx * (2 + ny * 2)] = 20.2; // Interior neighbor at i=nx-2
        apply_neumann_pressure(p, "x_max", nx, ny, nz, 1.0, 1.0, 1.0, 1.0, {0.0, 0.0, 0.0});
        int idx_boundary = (nx - 1) + nx * (2 + ny * 2);
        EXPECT_NEAR(p[idx_boundary], 20.2, 1e-12);
    }

    // Test y_min boundary
    {
        std::vector<double> p(total_cells, 0.0);
        p[2 + nx * (1 + ny * 2)] = 15.3; // Interior neighbor at j=1
        apply_neumann_pressure(p, "y_min", nx, ny, nz, 1.0, 1.0, 1.0, 1.0, {0.0, 0.0, 0.0});
        int idx_boundary = 2 + nx * (0 + ny * 2);
        EXPECT_NEAR(p[idx_boundary], 15.3, 1e-12);
    }

    // Test y_max boundary
    {
        std::vector<double> p(total_cells, 0.0);
        p[2 + nx * ((ny - 2) + ny * 2)] = 25.4; // Interior neighbor at j=ny-2
        apply_neumann_pressure(p, "y_max", nx, ny, nz, 1.0, 1.0, 1.0, 1.0, {0.0, 0.0, 0.0});
        int idx_boundary = 2 + nx * ((ny - 1) + ny * 2);
        EXPECT_NEAR(p[idx_boundary], 25.4, 1e-12);
    }

    // Test z_min boundary
    {
        std::vector<double> p(total_cells, 0.0);
        p[2 + nx * (2 + ny * 1)] = 30.1; // Interior neighbor at k=1
        apply_neumann_pressure(p, "z_min", nx, ny, nz, 1.0, 1.0, 1.0, 1.0, {0.0, 0.0, 0.0});
        int idx_boundary = 2 + nx * (2 + ny * 0);
        EXPECT_NEAR(p[idx_boundary], 30.1, 1e-12);
    }

    // Test z_max boundary
    {
        std::vector<double> p(total_cells, 0.0);
        p[2 + nx * (2 + ny * (nz - 2))] = 35.6; // Interior neighbor at k=nz-2
        apply_neumann_pressure(p, "z_max", nx, ny, nz, 1.0, 1.0, 1.0, 1.0, {0.0, 0.0, 0.0});
        int idx_boundary = 2 + nx * (2 + ny * (nz - 1));
        EXPECT_NEAR(p[idx_boundary], 35.6, 1e-12);
    }

    // Test composite "wall" boundary type across all faces
    {
        std::vector<double> p(total_cells, 0.0);
        p[1 + nx * (2 + ny * 2)] = 11.0; // x_min face ref
        p[(nx - 2) + nx * (2 + ny * 2)] = 12.0; // x_max face ref
        p[2 + nx * (1 + ny * 2)] = 13.0; // y_min face ref
        p[2 + nx * ((ny - 2) + ny * 2)] = 14.0; // y_max face ref
        p[2 + nx * (2 + ny * 1)] = 15.0; // z_min face ref
        p[2 + nx * (2 + ny * (nz - 2))] = 16.0; // z_max face ref

        apply_neumann_pressure(p, "wall", nx, ny, nz, 1.0, 1.0, 1.0, 1.0, {0.0, 0.0, 0.0});

        EXPECT_NEAR(p[0 + nx * (2 + ny * 2)], 11.0, 1e-12);
        EXPECT_NEAR(p[(nx - 1) + nx * (2 + ny * 2)], 12.0, 1e-12);
        EXPECT_NEAR(p[2 + nx * (0 + ny * 2)], 13.0, 1e-12);
        EXPECT_NEAR(p[2 + nx * ((ny - 1) + ny * 2)], 14.0, 1e-12);
        EXPECT_NEAR(p[2 + nx * (2 + ny * 0)], 15.0, 1e-12);
        EXPECT_NEAR(p[2 + nx * (2 + ny * (nz - 1))], 16.0, 1e-12);
    }
}

// ============================================================================
// NARRATIVE SECTION 2: Neumann Pressure Validation and Exception Checks
// ============================================================================
// Verifies that invalid grid configurations or incorrect gravity vector dimensions 
// throw appropriate exceptions during Neumann boundary application.
// ============================================================================

TEST(PressurePoissonTest, ApplyNeumannPressureValidationErrors) {
    int nx = 5, ny = 5, nz = 5;
    std::vector<double> p(nx * ny * nz, 0.0);

    // Invalid spacing (dx <= 0.0)
    EXPECT_THROW(
        apply_neumann_pressure(p, "x_min", nx, ny, nz, 0.0, 1.0, 1.0, 1.0, {0.0, 0.0, 0.0}),
        std::invalid_argument
    );

    // Invalid gravity vector size (!= 3)
    EXPECT_THROW(
        apply_neumann_pressure(p, "x_min", nx, ny, nz, 1.0, 1.0, 1.0, 1.0, {0.0, 0.0}),
        std::invalid_argument
    );
}

// ============================================================================
// NARRATIVE SECTION 3: Solid Internal Cell Pressure Averaging
// ============================================================================
// For immersed boundaries or internal solid regions where the mask value is 0,
// the pressure is interpolated from the 6 orthogonal neighbor cells to ensure 
// smoothness and satisfy discrete Neumann equilibrium:
//        p_{i,j,k} = (p_{i+1,j,k} + p_{i-1,j,k} + p_{i,j+1,k} + p_{i,j-1,k} + p_{i,j,k+1} + p_{i,j,k-1}) / 6
// ============================================================================

TEST(PressurePoissonTest, SolidNeumannPressureAveraging) {
    int nx = 5, ny = 5, nz = 5;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1); // 1 = fluid

    // Designate cell (2, 2, 2) as solid (mask == 0)
    int i = 2, j = 2, k = 2;
    int idx_solid = i + nx * (j + ny * k);
    mask[idx_solid] = 0;

    // Set neighbor pressures to known values
    p[(i - 1) + nx * (j + ny * k)] = 1.0; // west
    p[(i + 1) + nx * (j + ny * k)] = 3.0; // east
    p[i + nx * ((j - 1) + ny * k)] = 2.0; // south
    p[i + nx * ((j + 1) + ny * k)] = 4.0; // north
    p[i + nx * (j + ny * (k - 1))] = 5.0; // down
    p[i + nx * (j + ny * (k + 1))] = 6.0; // up

    double dx = 1.0, dy = 1.0, dz = 1.0;
    apply_solid_neumann_pressure_parallel(p, mask, nx, ny, nz, dx, dy, dz);

    // Expected average: (1 + 3 + 2 + 4 + 5 + 6) / 6 = 21 / 6 = 3.5
    EXPECT_NEAR(p[idx_solid], 3.5, 1e-12);
}

// ============================================================================
// NARRATIVE SECTION 4: Poisson Solver Validation and Error-Handling Paths
// ============================================================================
// Verifies that solve_poisson_red_black_parallel robustly catches invalid inputs
// such as small grid dimensions, negative spacing, invalid iteration bounds,
// and vector size mismatches.
// ============================================================================

TEST(PressurePoissonTest, PoissonSolverValidationErrors) {
    int nx = 5, ny = 5, nz = 5;
    std::vector<double> p(nx * ny * nz, 0.0);
    std::vector<double> rhs(nx * ny * nz, 0.0);
    std::vector<int> mask(nx * ny * nz, 1);
    std::vector<BoundaryCondition> bc_list;

    // Small grid dimensions (< 3x3x3)
    EXPECT_THROW(
        solve_poisson_red_black_parallel(p, rhs, mask, bc_list, 2, ny, nz, 1.0, 1.0, 1.0, 10, 1e-6, 1.0, {0.0, 0.0, 0.0}),
        std::invalid_argument
    );

    // Non-positive grid spacing
    EXPECT_THROW(
        solve_poisson_red_black_parallel(p, rhs, mask, bc_list, nx, ny, nz, 0.0, 1.0, 1.0, 10, 1e-6, 1.0, {0.0, 0.0, 0.0}),
        std::invalid_argument
    );

    // Invalid max iterations
    EXPECT_THROW(
        solve_poisson_red_black_parallel(p, rhs, mask, bc_list, nx, ny, nz, 1.0, 1.0, 1.0, 0, 1e-6, 1.0, {0.0, 0.0, 0.0}),
        std::invalid_argument
    );

    // Vector size mismatch
    std::vector<double> bad_p(10, 0.0);
    EXPECT_THROW(
        solve_poisson_red_black_parallel(bad_p, rhs, mask, bc_list, nx, ny, nz, 1.0, 1.0, 1.0, 10, 1e-6, 1.0, {0.0, 0.0, 0.0}),
        std::invalid_argument
    );
}

// ============================================================================
// NARRATIVE SECTION 5: Red-Black Gauss-Seidel Poisson Solver Execution & Explosion
// ============================================================================
// Verifies that the complete Red-Black parallel Poisson solver successfully 
// iterates over fluid cells, integrates source terms (rhs), and updates boundary 
// conditions. Also tests that non-finite (NaN/inf) values trigger an immediate 
// runtime explosion exception.
// ============================================================================

TEST(PressurePoissonTest, RedBlackPoissonSolverExecution) {
    int nx = 5, ny = 5, nz = 5;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> p(total_cells, 0.0);
    std::vector<double> rhs(total_cells, 0.1);
    std::vector<int> mask(total_cells, 1);

    // Create boundary conditions list containing a non-pressure type to trigger apply_neumann_pressure
    std::vector<BoundaryCondition> bc_list = {
        {"velocity", "x_min"},
        {"pressure", "x_max"}
    };

    double dx = 0.1, dy = 0.1, dz = 0.1;
    int max_iters = 10;
    double tol = 1e-6;

    EXPECT_NO_THROW(
        solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list,
            nx, ny, nz,
            dx, dy, dz,
            max_iters, tol,
            1.0, {0.0, 0.0, 0.0}
        )
    );

    // Verify that interior fluid cells were updated from initial zero state
    int interior_idx = 2 + nx * (2 + ny * 2);
    EXPECT_NE(p[interior_idx], 0.0);
}

TEST(PressurePoissonTest, RedBlackPoissonSolverExplosionHandling) {
    int nx = 5, ny = 5, nz = 5;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> p(total_cells, NAN); // Seed with NaN to trigger explosion path
    std::vector<double> rhs(total_cells, 0.1);
    std::vector<int> mask(total_cells, 1);
    std::vector<BoundaryCondition> bc_list;

    EXPECT_THROW(
        solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list,
            nx, ny, nz,
            0.1, 0.1, 0.1,
            5, 1e-6,
            1.0, {0.0, 0.0, 0.0}
        ),
        std::runtime_error
    );
}
