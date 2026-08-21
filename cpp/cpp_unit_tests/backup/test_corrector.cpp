/**
 * @file test_corrector.cpp
 * @brief Comprehensive Unit Test Suite for Step 4 Corrector Velocity Projection.
 * 
 * LITERATE TESTING NARRATIVE & MATHEMATICAL FORMULATION:
 * ---------------------------------------------------------------------------------
 * This test suite validates the correctness, robustness, and exception safety 
 * of the pressure projection corrector step in the Navier-Stokes solver.
 * 
 * The corrector projection updates trial velocities (u_star, v_star, w_star) 
 * to divergence-free velocities (u, v, w) using the pressure gradient:
 *     u^(n+1) = u* - (dt / rho) * (dp/dx)
 *     v^(n+1) = v* - (dt / rho) * (dp/dy)
 *     w^(n+1) = w* - (dt / rho) * (dp/dz)
 * 
 * We verify both nominal operational convergence and all defensive guard clauses:
 * 1. Grid dimension validation (nx, ny, nz >= 3)
 * 2. Grid spacing validation (dx, dy, dz > 0.0)
 * 3. Physics parameter validation (dt, rho > 0.0)
 * 4. Vector size contract validation (matching total_cells)
 * 5. Numerical stability / NaN handling (runtime error on non-finite values)
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cassert>
#include "corrector.hpp"

using namespace navier_stokes_solver;

TEST(CorrectorTest, NominalProjection) {
    // We define a 3x3x3 grid with unit spacing.
    int nx = 3;
    int ny = 3;
    int nz = 3;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
    double dt = 0.01;
    double rho = 1000.0;

    assert(nx >= 3 && ny >= 3 && nz >= 3);
    assert(dx > 0.0 && dy > 0.0 && dz > 0.0);
    assert(dt > 0.0 && rho > 0.0);

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 1.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    
    // We define a linear pressure field:
    //     p(i, j, k) = i * 100.0
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = static_cast<size_t>(i + nx * (j + ny * k));
                p[idx] = static_cast<double>(i) * 100.0;
            }
        }
    }

    // We execute the parallel corrector projection step.
    solve_corrector_parallel(u, v, w, u_star, v_star, w_star, p, mask, nx, ny, nz, dx, dy, dz, dt, rho);

    // We verify the corrected velocity at center cell (1, 1, 1):
    //     dp_dx = (p(2,1,1) - p(1,1,1)) / dx = (200 - 100) / 1.0 = 100.0
    //     coeff = dt / rho = 0.01 / 1000.0 = 1e-5
    //     new_u = u_star - coeff * dp_dx = 1.0 - (1e-5 * 100.0) = 1.0 - 0.001 = 0.999
    size_t center_idx = static_cast<size_t>(1 + nx * (1 + ny * 1));
    double expected_u = 1.0 - (dt / rho) * (100.0 / dx);

    assert(std::abs(u[center_idx] - expected_u) < 1e-9);
    ASSERT_NEAR(u[center_idx], expected_u, 1e-9);
}

TEST(CorrectorTest, InvalidGridDimensionsException) {
    // We test guard clause for grid dimensions < 3 (triggers line 32).
    int nx = 2; // Invalid (< 3)
    int ny = 3;
    int nz = 3;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
    double dt = 0.01;
    double rho = 1000.0;

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);

    // Expect std::invalid_argument for geometry error.
    ASSERT_THROW({
        solve_corrector_parallel(u, v, w, u_star, v_star, w_star, p, mask, nx, ny, nz, dx, dy, dz, dt, rho);
    }, std::invalid_argument);
}

TEST(CorrectorTest, InvalidGridSpacingException) {
    // We test guard clause for non-positive grid spacing (triggers line 35).
    int nx = 3;
    int ny = 3;
    int nz = 3;
    double dx = 0.0; // Invalid (<= 0.0)
    double dy = 1.0;
    double dz = 1.0;
    double dt = 0.01;
    double rho = 1000.0;

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);

    ASSERT_THROW({
        solve_corrector_parallel(u, v, w, u_star, v_star, w_star, p, mask, nx, ny, nz, dx, dy, dz, dt, rho);
    }, std::invalid_argument);
}

TEST(CorrectorTest, InvalidPhysicsParametersException) {
    // We test guard clause for non-positive time step or density (triggers line 38).
    int nx = 3;
    int ny = 3;
    int nz = 3;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
    double dt = -0.01; // Invalid (<= 0.0)
    double rho = 1000.0;

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);

    ASSERT_THROW({
        solve_corrector_parallel(u, v, w, u_star, v_star, w_star, p, mask, nx, ny, nz, dx, dy, dz, dt, rho);
    }, std::invalid_argument);
}

TEST(CorrectorTest, VectorSizeMismatchException) {
    // We test guard clause for vector size mismatch (triggers line 45).
    int nx = 3;
    int ny = 3;
    int nz = 3;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
    double dt = 0.01;
    double rho = 1000.0;

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    std::vector<double> u(total_cells - 1, 0.0); // Size mismatch
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);

    ASSERT_THROW({
        solve_corrector_parallel(u, v, w, u_star, v_star, w_star, p, mask, nx, ny, nz, dx, dy, dz, dt, rho);
    }, std::invalid_argument);
}

TEST(CorrectorTest, NonFiniteVelocityException) {
    // We test forensic numerical audit and runtime exception on non-finite values (triggers lines 122-125).
    int nx = 3;
    int ny = 3;
    int nz = 3;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
    double dt = 0.01;
    double rho = 1000.0;

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, NAN); // Induce non-finite velocity
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);

    ASSERT_THROW({
        solve_corrector_parallel(u, v, w, u_star, v_star, w_star, p, mask, nx, ny, nz, dx, dy, dz, dt, rho);
    }, std::runtime_error);
}
