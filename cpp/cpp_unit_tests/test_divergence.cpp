/**
 * @file test_divergence.cpp
 * @brief Literate test suite for the 3D Divergence Operator with Multi-Threading Verification.
 * 
 * LITERATE TESTING NARRATIVE & MATHEMATICAL FORMULATION:
 * ---------------------------------------------------------------------------------
 * This test file narrates and verifies the analytical accuracy, multi-threading 
 * execution correctness, geometry safety, and numerical exception handling of the 
 * C++ compute_divergence kernel used for the Pressure Poisson Equation (PPE) source term.
 * 
 * The divergence of a velocity vector field u_star = (u_star, v_star, w_star) is defined as:
 *     div(u_star) = du_star/dx + dv_star/dy + dw_star/dz
 * 
 * Using second-order central differences:
 *     du_star/dx = (u_star(i+1, j, k) - u_star(i-1, j, k)) / (2 * dx)
 *     dv_star/dy = (v_star(i, j+1, k) - v_star(i, j-1, k)) / (2 * dy)
 *     dw_star/dz = (w_star(i, j, k+1) - w_star(i, j, k-1)) / (2 * dz)
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cassert>
#include "divergence.hpp"
#include "grid_math.hpp"

using namespace navier_stokes_solver;

class DivergenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // We define standard grid dimensions and uniform spatial steps for our test domain.
        Nx = 5;
        Ny = 5;
        Nz = 5;
        dx = 0.1;
        dy = 0.1;
        dz = 0.1;
    }

    int Nx, Ny, Nz;
    double dx, dy, dz;
};

/**
 * Test Case 1: Analytical Linear Velocity Field Divergence Exactness
 * 
 * We construct velocity vector components following linear distributions:
 *     u_star(x, y, z) = 2.0 * x
 *     v_star(x, y, z) = -3.0 * y
 *     w_star(x, y, z) = 4.0 * z
 * 
 * Using second-order central differences, the analytical partial derivatives are:
 *     du_star/dx = 2.0
 *     dv_star/dy = -3.0
 *     dw_star/dz = 4.0
 * 
 * The total scalar divergence is their sum:
 *     div(u_star) = du_star/dx + dv_star/dy + dw_star/dz = 2.0 + (-3.0) + 4.0 = 3.0
 */
TEST_F(DivergenceTest, LinearFieldExactDivergence) {
    size_t total_size = static_cast<size_t>(Nx) * Ny * Nz;
    std::vector<double> u_star(total_size, 0.0);
    std::vector<double> v_star(total_size, 0.0);
    std::vector<double> w_star(total_size, 0.0);
    std::vector<double> div_out(total_size, 0.0);

    assert(Nx > 2 && Ny > 2 && Nz > 2);
    assert(dx > 0.0 && dy > 0.0 && dz > 0.0);

    // We populate the velocity vector fields using our linear analytical formulas.
    for (int i = 0; i < Nx; ++i) {
        double x = i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = j * dy;
            for (int k = 0; k < Nz; ++k) {
                double z = k * dz;
                size_t idx = static_cast<size_t>(get_flat_index(i, j, k, Nx, Ny));
                
                u_star[idx] = 2.0 * x;
                v_star[idx] = -3.0 * y;
                w_star[idx] = 4.0 * z;
            }
        }
    }

    // We execute the C++ divergence operator kernel.
    compute_divergence(u_star.data(), v_star.data(), w_star.data(), div_out.data(), Nx, Ny, Nz, dx, dy, dz);

    // We assert that interior node divergence values match the analytical sum (3.0) within machine precision.
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t idx = static_cast<size_t>(get_flat_index(i, j, k, Nx, Ny));
                assert(std::abs(div_out[idx] - 3.0) < 1e-9);
                EXPECT_NEAR(div_out[idx], 3.0, 1e-9);
            }
        }
    }
}

/**
 * Test Case 2: Multi-Threading Parallel Execution Correctness
 * 
 * To ensure OpenMP multi-threading executes correctly without data corruption, 
 * synchronization bugs, or race conditions during parallel traversal, we instantiate 
 * a large grid domain of size 15 x 15 x 15 = 3375 cells. This strictly exceeds our 
 * execution threshold of 1000 cells, activating the OpenMP parallel loops across multiple CPU cores.
 * 
 * Using the linear velocity fields:
 *     u_star(x, y, z) = 2.0 * x
 *     v_star(x, y, z) = -3.0 * y
 *     w_star(x, y, z) = 4.0 * z
 * 
 * The computed parallel result across all threads must identically match 
 * the exact analytical constant sum of 3.0.
 */
TEST_F(DivergenceTest, MultiThreadingParallelExecutionCorrectness) {
    int large_Nx = 15;
    int large_Ny = 15;
    int large_Nz = 15;
    size_t total_size = static_cast<size_t>(large_Nx) * large_Ny * large_Nz;

    std::vector<double> u_star(total_size, 0.0);
    std::vector<double> v_star(total_size, 0.0);
    std::vector<double> w_star(total_size, 0.0);
    std::vector<double> div_out(total_size, 0.0);

    // Populate velocity fields on the large multi-threaded grid
    for (int i = 0; i < large_Nx; ++i) {
        double x = i * dx;
        for (int j = 0; j < large_Ny; ++j) {
            double y = j * dy;
            for (int k = 0; k < large_Nz; ++k) {
                double z = k * dz;
                size_t idx = static_cast<size_t>(get_flat_index(i, j, k, large_Nx, large_Ny));
                
                u_star[idx] = 2.0 * x;
                v_star[idx] = -3.0 * y;
                w_star[idx] = 4.0 * z;
            }
        }
    }

    // Execute multi-threaded divergence kernel safely without throwing exceptions
    EXPECT_NO_THROW({
        compute_divergence(u_star.data(), v_star.data(), w_star.data(), div_out.data(), 
                          large_Nx, large_Ny, large_Nz, dx, dy, dz);
    });

    // Verify that every interior cell processed in parallel matches the exact analytical solution
    for (int i = 1; i < large_Nx - 1; ++i) {
        for (int j = 1; j < large_Ny - 1; ++j) {
            for (int k = 1; k < large_Nz - 1; ++k) {
                size_t idx = static_cast<size_t>(get_flat_index(i, j, k, large_Nx, large_Ny));
                EXPECT_NEAR(div_out[idx], 3.0, 1e-9);
                EXPECT_TRUE(std::isfinite(div_out[idx]));
            }
        }
    }
}

/**
 * Test Case 3: Geometry Guard Verification
 * 
 * Providing zero or negative grid spacing (e.g., dx = 0.0) represents an invalid physical 
 * space configuration and must trigger an invalid_argument exception.
 */
TEST_F(DivergenceTest, ZeroGridSpacingThrows) {
    size_t total_size = static_cast<size_t>(Nx) * Ny * Nz;
    std::vector<double> u_star(total_size, 1.0);
    std::vector<double> v_star(total_size, 1.0);
    std::vector<double> w_star(total_size, 1.0);
    std::vector<double> div_out(total_size, 0.0);

    double invalid_dx = 0.0; // Invalid grid spacing trigger

    // Supplying zero grid spacing should invoke the geometry guard.
    EXPECT_THROW({
        compute_divergence(u_star.data(), v_star.data(), w_star.data(), div_out.data(), Nx, Ny, Nz, invalid_dx, dy, dz);
    }, std::invalid_argument);
}

/**
 * Test Case 4: Non-Finite Numerical Audit
 * 
 * If a non-finite value (such as infinity or NaN) is introduced into any velocity 
 * component field, the numerical audit mechanism must intercept it and throw a runtime_error safely across thread boundaries.
 */
TEST_F(DivergenceTest, NonFiniteVelocityFieldThrows) {
    size_t total_size = static_cast<size_t>(Nx) * Ny * Nz;
    std::vector<double> u_star(total_size, 1.0);
    std::vector<double> v_star(total_size, 1.0);
    std::vector<double> w_star(total_size, 1.0);
    std::vector<double> div_out(total_size, 0.0);

    // We inject an infinity into a target cell of the u_star field.
    size_t target_idx = static_cast<size_t>(get_flat_index(2, 2, 2, Nx, Ny));
    u_star[target_idx] = std::numeric_limits<double>::infinity();

    // The forensic numeric checker should catch the resulting non-finite divergence value.
    EXPECT_THROW({
        compute_divergence(u_star.data(), v_star.data(), w_star.data(), div_out.data(), Nx, Ny, Nz, dx, dy, dz);
    }, std::runtime_error);
}
