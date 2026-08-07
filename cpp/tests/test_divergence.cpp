/**
 * @file test_divergence.cpp
 * @brief Literate test suite for the 3D Divergence Operator.
 * 
 * This test file narrates and verifies the analytical accuracy, geometry safety,
 * and numerical exception handling of the C++ compute_divergence kernel used 
 * for the Pressure Poisson Equation (PPE) source term.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "divergence.hpp"

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
 *     u*(x, y, z) = 2.0 * x
 *     v*(x, y, z) = -3.0 * y
 *     w*(x, y, z) = 4.0 * z
 * 
 * Using second-order central differences, the analytical partial derivatives are:
 *     du*/dx = 2.0
 *     dv*/dy = -3.0
 *     dw*/dz = 4.0
 * 
 * The total scalar divergence is their sum:
 *     div u* = du*/dx + dv*/dy + dw*/dz = 2.0 + (-3.0) + 4.0 = 3.0
 */
TEST_F(DivergenceTest, LinearFieldExactDivergence) {
    size_t total_size = static_cast<size_t>(Nx) * Ny * Nz;
    std::vector<double> u_star(total_size, 0.0);
    std::vector<double> v_star(total_size, 0.0);
    std::vector<double> w_star(total_size, 0.0);
    std::vector<double> div_out(total_size, 0.0);

    // We populate the velocity vector fields using our linear analytical formulas.
    for (int i = 0; i < Nx; ++i) {
        double x = i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = j * dy;
            for (int k = 0; k < Nz; ++k) {
                double z = k * dz;
                size_t idx = static_cast<size_t>(i) * (Ny * Nz) + static_cast<size_t>(j) * Nz + k;
                
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
                size_t idx = static_cast<size_t>(i) * (Ny * Nz) + static_cast<size_t>(j) * Nz + k;
                EXPECT_NEAR(div_out[idx], 3.0, 1e-9);
            }
        }
    }
}

/**
 * Test Case 2: Geometry Guard Verification
 * 
 * Providing zero grid spacing (e.g., dx = 0.0) represents an invalid physical 
 * space configuration and must trigger an invalid_argument exception.
 */
TEST_F(DivergenceTest, ZeroGridSpacingThrows) {
    std::vector<double> u_star(Nx * Ny * Nz, 1.0);
    std::vector<double> v_star(Nx * Ny * Nz, 1.0);
    std::vector<double> w_star(Nx * Ny * Nz, 1.0);
    std::vector<double> div_out(Nx * Ny * Nz, 0.0);

    // Supplying zero grid spacing should invoke the geometry guard.
    EXPECT_THROW({
        compute_divergence(u_star.data(), v_star.data(), w_star.data(), div_out.data(), Nx, Ny, Nz, 0.0, dy, dz);
    }, std::invalid_argument);
}

/**
 * Test Case 3: Non-Finite Numerical Audit
 * 
 * If a non-finite value (such as infinity or NaN) is introduced into any velocity 
 * component field, the numerical audit mechanism must intercept it and throw a runtime_error.
 */
TEST_F(DivergenceTest, NonFiniteVelocityFieldThrows) {
    std::vector<double> u_star(Nx * Ny * Nz, 1.0);
    std::vector<double> v_star(Nx * Ny * Nz, 1.0);
    std::vector<double> w_star(Nx * Ny * Nz, 1.0);
    std::vector<double> div_out(Nx * Ny * Nz, 0.0);

    // We inject an infinity into a target cell of the u_star field.
    size_t target_idx = static_cast<size_t>(2) * (Ny * Nz) + static_cast<size_t>(2) * Nz + 2;
    u_star[target_idx] = __builtin_inf();

    // The forensic numeric checker should catch the resulting non-finite divergence value.
    EXPECT_THROW({
        compute_divergence(u_star.data(), v_star.data(), w_star.data(), div_out.data(), Nx, Ny, Nz, dx, dy, dz);
    }, std::runtime_error);
}
