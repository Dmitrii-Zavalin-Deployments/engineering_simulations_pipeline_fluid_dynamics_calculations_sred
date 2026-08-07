/**
 * @file test_gradient.cpp
 * @brief Literate test suite for the 3D Gradient Operator with Multi-Threading Verification.
 * 
 * This test file narrates and verifies the analytical accuracy, multi-threading 
 * execution correctness, geometry safety, and numerical exception handling of the 
 * C++ compute_gradient kernel.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "gradient.hpp"

using namespace ops;

class GradientTest : public ::testing::Test {
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
 * Test Case 1: Analytical Linear Gradient Exactness
 * 
 * We construct a scalar pressure field following a linear distribution:
 *     p(x, y, z) = 3.0 * x - 2.0 * y + 5.0 * z
 * 
 * Using second-order central differences, the computed derivative components 
 * should match the exact analytical slopes across the interior domain:
 *     ∂p/∂x = 3.0,  ∂p/∂y = -2.0,  ∂p/∂z = 5.0
 */
TEST_F(GradientTest, LinearFieldExactDerivatives) {
    size_t total_size = static_cast<size_t>(Nx) * Ny * Nz;
    std::vector<double> field(total_size, 0.0);
    std::vector<double> grad_x(total_size, 0.0);
    std::vector<double> grad_y(total_size, 0.0);
    std::vector<double> grad_z(total_size, 0.0);

    // We populate the field grid using our linear analytical formula.
    for (int i = 0; i < Nx; ++i) {
        double x = i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = j * dy;
            for (int k = 0; k < Nz; ++k) {
                double z = k * dz;
                size_t idx = static_cast<size_t>(i) * (Ny * Nz) + static_cast<size_t>(j) * Nz + k;
                field[idx] = 3.0 * x - 2.0 * y + 5.0 * z;
            }
        }
    }

    // We execute the C++ gradient operator kernel.
    compute_gradient(field.data(), grad_x.data(), grad_y.data(), grad_z.data(), Nx, Ny, Nz, dx, dy, dz);

    // We assert that interior node gradients match analytical expectations within machine precision.
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t idx = static_cast<size_t>(i) * (Ny * Nz) + static_cast<size_t>(j) * Nz + k;
                EXPECT_NEAR(grad_x[idx], 3.0, 1e-9);
                EXPECT_NEAR(grad_y[idx], -2.0, 1e-9);
                EXPECT_NEAR(grad_z[idx], 5.0, 1e-9);
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
 * Using the linear scalar field:
 *     p(x, y, z) = 3.0 * x - 2.0 * y + 5.0 * z
 * 
 * The computed parallel gradient components across all threads must identically match 
 * the exact analytical slopes: 3.0, -2.0, and 5.0 respectively.
 */
TEST_F(GradientTest, MultiThreadingParallelExecutionCorrectness) {
    int large_Nx = 15;
    int large_Ny = 15;
    int large_Nz = 15;
    size_t total_size = static_cast<size_t>(large_Nx) * large_Ny * large_Nz;

    std::vector<double> field(total_size, 0.0);
    std::vector<double> grad_x(total_size, 0.0);
    std::vector<double> grad_y(total_size, 0.0);
    std::vector<double> grad_z(total_size, 0.0);

    // Populate the field grid on the large multi-threaded domain
    for (int i = 0; i < large_Nx; ++i) {
        double x = i * dx;
        for (int j = 0; j < large_Ny; ++j) {
            double y = j * dy;
            for (int k = 0; k < large_Nz; ++k) {
                double z = k * dz;
                size_t idx = static_cast<size_t>(i) * (large_Ny * large_Nz) + static_cast<size_t>(j) * large_Nz + k;
                field[idx] = 3.0 * x - 2.0 * y + 5.0 * z;
            }
        }
    }

    // Execute multi-threaded gradient kernel safely without throwing exceptions
    EXPECT_NO_THROW({
        compute_gradient(field.data(), grad_x.data(), grad_y.data(), grad_z.data(), 
                          large_Nx, large_Ny, large_Nz, dx, dy, dz);
    });

    // Verify that every interior cell processed in parallel matches the exact analytical solution
    for (int i = 1; i < large_Nx - 1; ++i) {
        for (int j = 1; j < large_Ny - 1; ++j) {
            for (int k = 1; k < large_Nz - 1; ++k) {
                size_t idx = static_cast<size_t>(i) * (large_Ny * large_Nz) + static_cast<size_t>(j) * large_Nz + k;
                EXPECT_NEAR(grad_x[idx], 3.0, 1e-9);
                EXPECT_NEAR(grad_y[idx], -2.0, 1e-9);
                EXPECT_NEAR(grad_z[idx], 5.0, 1e-9);
                EXPECT_TRUE(std::isfinite(grad_x[idx]));
                EXPECT_TRUE(std::isfinite(grad_y[idx]));
                EXPECT_TRUE(std::isfinite(grad_z[idx]));
            }
        }
    }
}

/**
 * Test Case 3: Geometry Guard Verification
 * 
 * Providing non-positive grid steps (e.g., Δx <= 0) represents an invalid physical 
 * space configuration and must trigger an invalid_argument exception.
 */
TEST_F(GradientTest, InvalidGridSpacingThrows) {
    std::vector<double> field(Nx * Ny * Nz, 1.0);
    std::vector<double> grad_x(Nx * Ny * Nz, 0.0);
    std::vector<double> grad_y(Nx * Ny * Nz, 0.0);
    std::vector<double> grad_z(Nx * Ny * Nz, 0.0);

    // Supplying a negative grid spacing increment should invoke the geometry guard.
    EXPECT_THROW({
        compute_gradient(field.data(), grad_x.data(), grad_y.data(), grad_z.data(), Nx, Ny, Nz, -0.1, dy, dz);
    }, std::invalid_argument);
}

/**
 * Test Case 4: Non-Finite Numerical Audit
 * 
 * If a non-finite value (such as infinity or NaN) is introduced into the scalar 
 * field, the numerical audit mechanism must intercept it and throw a runtime_error safely across thread boundaries.
 */
TEST_F(GradientTest, NonFiniteFieldThrows) {
    std::vector<double> field(Nx * Ny * Nz, 1.0);
    std::vector<double> grad_x(Nx * Ny * Nz, 0.0);
    std::vector<double> grad_y(Nx * Ny * Nz, 0.0);
    std::vector<double> grad_z(Nx * Ny * Nz, 0.0);

    // We inject an infinity into a target cell.
    size_t target_idx = static_cast<size_t>(2) * (Ny * Nz) + static_cast<size_t>(2) * Nz + 2;
    field[target_idx] = __builtin_inf();

    // The forensic numeric checker should catch the resulting non-finite gradient component.
    EXPECT_THROW({
        compute_gradient(field.data(), grad_x.data(), grad_y.data(), grad_z.data(), Nx, Ny, Nz, dx, dy, dz);
    }, std::runtime_error);
}
