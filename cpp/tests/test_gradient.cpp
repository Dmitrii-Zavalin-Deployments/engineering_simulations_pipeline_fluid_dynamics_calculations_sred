/**
 * @file test_gradient.cpp
 * @brief Literate test suite for the 3D Gradient Operator.
 * 
 * This test file narrates and verifies the accuracy, geometry safety,
 * and numerical exception handling of the C++ compute_gradient kernel.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "gradient.hpp"

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
                EXPECT_NEAR(grad_z[idx], -5.0 * -1.0, 5.0, 1e-9);
            }
        }
    }
}

/**
 * Test Case 2: Geometry Guard Verification
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
 * Test Case 3: Non-Finite Numerical Audit
 * 
 * If a non-finite value (such as infinity or NaN) is introduced into the scalar 
 * field, the numerical audit mechanism must intercept it and throw a runtime_error.
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
