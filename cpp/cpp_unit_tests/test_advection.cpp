/**
 * @file test_advection.cpp
 * @brief Literate test suite for the 3D Advection Operator with Multi-Threading Verification.
 * 
 * This test file narrates and verifies the analytical accuracy, multi-threading 
 * execution correctness, and numerical exception handling of the C++ compute_advection 
 * kernel using central differencing for the advective derivative term: (v · ∇)f.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "advection.hpp"
#include "grid_math.hpp"

using namespace navier_stokes_solver;

class AdvectionTest : public ::testing::Test {
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
 * Test Case 1: Analytical Linear Field Advection Exactness
 * 
 * We define a linear scalar field:
 *     f(x, y, z) = 2.0 * x + 3.0 * y + 4.0 * z
 * 
 * The analytical spatial gradients are uniform across the domain:
 *     ∂f/∂x = 2.0
 *     ∂f/∂y = 3.0
 *     ∂f/∂z = 4.0
 * 
 * We set uniform velocity components:
 *     u(x, y, z) = 1.0
 *     v(x, y, z) = 2.0
 *     w(x, y, z) = 3.0
 * 
 * The expected advection term is computed as:
 *     (v · ∇)f = u * (∂f/∂x) + v * (∂f/∂y) + w * (∂f/∂z)
 *              = (1.0 * 2.0) + (2.0 * 3.0) + (3.0 * 4.0)
 *              = 2.0 + 6.0 + 12.0 = 20.0
 */
TEST_F(AdvectionTest, LinearFieldExactAdvection) {
    size_t total_size = static_cast<size_t>(Nx) * Ny * Nz;
    std::vector<double> u(total_size, 1.0);
    std::vector<double> v(total_size, 2.0);
    std::vector<double> w(total_size, 3.0);
    std::vector<double> field(total_size, 0.0);
    std::vector<double> adv_out(total_size, 0.0);

    // We populate the scalar field grid using our linear analytical formula.
    for (int i = 0; i < Nx; ++i) {
        double x = i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = j * dy;
            for (int k = 0; k < Nz; ++k) {
                double z = k * dz;
                size_t idx = static_cast<size_t>(i) * (Ny * Nz) + static_cast<size_t>(j) * Nz + k;
                field[idx] = 2.0 * x + 3.0 * y + 4.0 * z;
            }
        }
    }

    // We execute the C++ advection operator kernel.
    compute_advection(u.data(), v.data(), w.data(), field.data(), adv_out.data(), Nx, Ny, Nz, dx, dy, dz);

    // We assert that interior node advection values match the analytical constant (20.0) within machine precision.
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t idx = static_cast<size_t>(i) * (Ny * Nz) + static_cast<size_t>(j) * Nz + k;
                EXPECT_NEAR(adv_out[idx], 20.0, 1e-9);
            }
        }
    }
}

/**
 * Test Case 2: Multi-Threading Parallel Execution Correctness
 * 
 * To ensure OpenMP multi-threading executes correctly without data corruption or race conditions, 
 * we construct a large domain of size 15 x 15 x 15 = 3375 cells. This strictly exceeds our 
 * execution threshold of 1000 cells, activating the OpenMP parallel loops across multiple CPU cores.
 * 
 * Using the same linear analytical field:
 *     f(x, y, z) = 2.0 * x + 3.0 * y + 4.0 * z
 *     u = 1.0, v = 2.0, w = 3.0
 * 
 * The expected parallel computed result across all threads must identically match 
 * the analytical constant value of 20.0.
 */
TEST_F(AdvectionTest, MultiThreadingParallelExecutionCorrectness) {
    int large_Nx = 15;
    int large_Ny = 15;
    int large_Nz = 15;
    size_t total_size = static_cast<size_t>(large_Nx) * large_Ny * large_Nz;

    std::vector<double> u(total_size, 1.0);
    std::vector<double> v(total_size, 2.0);
    std::vector<double> w(total_size, 3.0);
    std::vector<double> field(total_size, 0.0);
    std::vector<double> adv_out(total_size, 0.0);

    // Populate scalar field on the large multi-threaded grid
    for (int i = 0; i < large_Nx; ++i) {
        double x = i * dx;
        for (int j = 0; j < large_Ny; ++j) {
            double y = j * dy;
            for (int k = 0; k < large_Nz; ++k) {
                double z = k * dz;
                size_t idx = static_cast<size_t>(i) * (large_Ny * large_Nz) + static_cast<size_t>(j) * large_Nz + k;
                field[idx] = 2.0 * x + 3.0 * y + 4.0 * z;
            }
        }
    }

    // Execute multi-threaded advection kernel safely without throwing exceptions
    EXPECT_NO_THROW({
        compute_advection(u.data(), v.data(), w.data(), field.data(), adv_out.data(), 
                          large_Nx, large_Ny, large_Nz, dx, dy, dz);
    });

    // Verify that every interior cell processed in parallel matches the exact analytical solution
    for (int i = 1; i < large_Nx - 1; ++i) {
        for (int j = 1; j < large_Ny - 1; ++j) {
            for (int k = 1; k < large_Nz - 1; ++k) {
                size_t idx = static_cast<size_t>(i) * (large_Ny * large_Nz) + static_cast<size_t>(j) * large_Nz + k;
                EXPECT_NEAR(adv_out[idx], 20.0, 1e-9);
                EXPECT_TRUE(std::isfinite(adv_out[idx]));
            }
        }
    }
}

/**
 * Test Case 3: Non-Finite Numerical Audit
 * 
 * If a non-finite value (such as infinity or NaN) is introduced into the velocity 
 * or scalar field, the forensic numerical audit mechanism must intercept it and 
 * throw a runtime_error safely across thread boundaries.
 */
TEST_F(AdvectionTest, NonFiniteVelocityFieldThrows) {
    size_t total_size = static_cast<size_t>(Nx) * Ny * Nz;
    std::vector<double> u(total_size, 1.0);
    std::vector<double> v(total_size, 2.0);
    std::vector<double> w(total_size, 3.0);
    std::vector<double> field(total_size, 1.0);
    std::vector<double> adv_out(total_size, 0.0);

    // We inject an infinity into a target cell of the u-velocity component.
    size_t target_idx = static_cast<size_t>(2) * (Ny * Nz) + static_cast<size_t>(2) * Nz + 2;
    u[target_idx] = __builtin_inf();

    // The forensic numeric checker should catch the resulting non-finite advection value.
    EXPECT_THROW({
        compute_advection(u.data(), v.data(), w.data(), field.data(), adv_out.data(), Nx, Ny, Nz, dx, dy, dz);
    }, std::runtime_error);
}
