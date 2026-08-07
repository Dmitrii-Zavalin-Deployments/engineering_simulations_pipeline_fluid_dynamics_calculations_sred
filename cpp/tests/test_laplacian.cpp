/**
 * @file test_laplacian.cpp
 * @brief Literate test suite for the 3D Discrete Laplacian Operator.
 * 
 * This test file narrates and verifies the analytical accuracy, geometry safety,
 * and numerical exception handling of the C++ compute_laplacian kernel using
 * a 7-point central difference stencil.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "laplacian.hpp"

class LaplacianTest : public ::testing::Test {
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
 * Test Case 1: Analytical Quadratic Field Laplacian Exactness
 * 
 * We construct a scalar field following a quadratic distribution:
 *     f(x, y, z) = 1.0 * x^2 - 2.0 * y^2 + 3.0 * z^2
 * 
 * Using second-order central differences for the 7-point Laplacian stencil, 
 * the second partial derivatives are constant across the interior domain:
 *     d2f/dx2 = 2.0
 *     d2f/dy2 = -4.0
 *     d2f/dz2 = 6.0
 * 
 * The total discrete Laplacian is their sum:
 *     laplacian = d2f/dx2 + d2f/dy2 + d2f/dz2 = 2.0 + (-4.0) + 6.0 = 4.0
 */
TEST_F(LaplacianTest, QuadraticFieldExactLaplacian) {
    size_t total_size = static_cast<size_t>(Nx) * Ny * Nz;
    std::vector<double> field(total_size, 0.0);
    std::vector<double> lap_out(total_size, 0.0);

    // We populate the field grid using our quadratic analytical formula.
    for (int i = 0; i < Nx; ++i) {
        double x = i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = j * dy;
            for (int k = 0; k < Nz; ++k) {
                double z = k * dz;
                size_t idx = static_cast<size_t>(i) * (Ny * Nz) + static_cast<size_t>(j) * Nz + k;
                field[idx] = 1.0 * (x * x) - 2.0 * (y * y) + 3.0 * (z * z);
            }
        }
    }

    // We execute the C++ Laplacian operator kernel.
    compute_laplacian(field.data(), lap_out.data(), Nx, Ny, Nz, dx, dy, dz);

    // We assert that interior node Laplacian values match the analytical sum (4.0) within machine precision.
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t idx = static_cast<size_t>(i) * (Ny * Nz) + static_cast<size_t>(j) * Nz + k;
                EXPECT_NEAR(lap_out[idx], 4.0, 1e-9);
            }
        }
    }
}

/**
 * Test Case 2: Geometry Guard Verification
 * 
 * Providing non-positive grid steps (e.g., dx <= 0.0) represents an invalid physical 
 * space configuration and must trigger an invalid_argument exception.
 */
TEST_F(LaplacianTest, InvalidGridSpacingThrows) {
    std::vector<double> field(Nx * Ny * Nz, 1.0);
    std::vector<double> lap_out(Nx * Ny * Nz, 0.0);

    // Supplying a negative grid spacing increment should invoke the geometry guard.
    EXPECT_THROW({
        compute_laplacian(field.data(), lap_out.data(), Nx, Ny, Nz, -0.1, dy, dz);
    }, std::invalid_argument);
}

/**
 * Test Case 3: Non-Finite Numerical Audit
 * 
 * If a non-finite value (such as infinity or NaN) is introduced into the scalar 
 * field, the numerical audit mechanism must intercept it and throw a runtime_error.
 */
TEST_F(LaplacianTest, NonFiniteFieldThrows) {
    std::vector<double> field(Nx * Ny * Nz, 1.0);
    std::vector<double> lap_out(Nx * Ny * Nz, 0.0);

    // We inject an infinity into a target cell.
    size_t target_idx = static_cast<size_t>(2) * (Ny * Nz) + static_cast<size_t>(2) * Nz + 2;
    field[target_idx] = __builtin_inf();

    // The forensic numeric checker should catch the resulting non-finite Laplacian value.
    EXPECT_THROW({
        compute_laplacian(field.data(), lap_out.data(), Nx, Ny, Nz, dx, dy, dz);
    }, std::runtime_error);
}
