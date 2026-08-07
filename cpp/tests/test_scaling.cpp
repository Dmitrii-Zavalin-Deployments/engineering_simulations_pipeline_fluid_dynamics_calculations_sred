/**
 * @file test_scaling.cpp
 * @brief Literate test suite for Navier-Stokes Scaling Factors.
 * 
 * This test file narrates and verifies the analytical accuracy, vacuum/zero 
 * safety guards, and numerical exception handling of the C++ scaling factor 
 * kernels: Δt / ρ and ρ / Δt.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include "scaling.hpp"

class ScalingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // We define standard fluid properties for our test domain.
        dt = 0.02;     // Time step (s)
        rho = 1.225;   // Fluid density (kg/m³)
    }

    double dt;
    double rho;
};

/**
 * Test Case 1: Predictor/Corrector Scaling Factor Exactness (Δt / ρ)
 * 
 * For a time step Δt = 0.02 s and density ρ = 1.225 kg/m³, we compute:
 *     scaling = Δt / ρ = 0.02 / 1.225 ≈ 0.0163265306122449
 */
TEST_F(ScalingTest, DtOverRhoValidComputation) {
    // We compute the predictor/corrector scaling factor.
    double result = get_dt_over_rho(dt, rho);

    // The expected value is:
    //     expected = 0.02 / 1.225
    double expected = dt / rho;

    // We assert that the computed scaling matches the analytical value within machine precision.
    EXPECT_NEAR(result, expected, 1e-12);
}

/**
 * Test Case 2: Vacuum/Negative Density Guard Verification (Δt / ρ)
 * 
 * Providing a zero or negative density (ρ <= 0.0) represents an unphysical 
 * vacuum state and must trigger an invalid_argument exception.
 */
TEST_F(ScalingTest, DtOverRhoInvalidDensityThrows) {
    // Supplying a zero or negative density should invoke the contract violation guard.
    EXPECT_THROW({
        get_dt_over_rho(dt, 0.0);
    }, std::invalid_argument);

    EXPECT_THROW({
        get_dt_over_rho(dt, -1.225);
    }, std::invalid_argument);
}

/**
 * Test Case 3: Non-Finite Numerical Audit for Predictor/Corrector Scaling
 * 
 * If a non-finite value (such as infinity) is introduced into the time step, 
 * the forensic numerical audit mechanism must intercept it and throw a runtime_error.
 */
TEST_F(ScalingTest, DtOverRhoNonFiniteThrows) {
    // We supply an infinite time-step to trigger the non-finite safety audit.
    double inf_dt = __builtin_inf();

    EXPECT_THROW({
        get_dt_over_rho(inf_dt, rho);
    }, std::runtime_error);
}

/**
 * Test Case 4: Pressure Poisson Equation Scaling Factor Exactness (ρ / Δt)
 * 
 * For density ρ = 1.225 kg/m³ and time step Δt = 0.02 s, we compute:
 *     scaling = ρ / Δt = 1.225 / 0.02 = 61.25
 */
TEST_F(ScalingTest, RhoOverDtValidComputation) {
    // We compute the pressure Poisson equation scaling factor.
    double result = get_rho_over_dt(dt, rho);

    // The expected value is:
    //     expected = 1.225 / 0.02 = 61.25
    double expected = rho / dt;

    // We assert that the computed scaling matches the analytical value within machine precision.
    EXPECT_NEAR(result, expected, 1e-12);
}

/**
 * Test Case 5: Zero/Negative Time-Step Guard Verification (ρ / Δt)
 * 
 * Providing a zero or negative time-step (Δt <= 0.0) represents an invalid 
 * temporal configuration and must trigger an invalid_argument exception.
 */
TEST_F(ScalingTest, RhoOverDtInvalidTimeStepThrows) {
    // Supplying a zero or negative time-step should invoke the temporal crash guard.
    EXPECT_THROW({
        get_rho_over_dt(0.0, rho);
    }, std::invalid_argument);

    EXPECT_THROW({
        get_rho_over_dt(-0.02, rho);
    }, std::invalid_argument);
}

/**
 * Test Case 6: Non-Finite Numerical Audit for Pressure Poisson Scaling
 * 
 * If a non-finite value (such as infinity) is introduced into the density, 
 * the forensic numerical audit mechanism must intercept it and throw a runtime_error.
 */
TEST_F(ScalingTest, RhoOverDtNonFiniteThrows) {
    // We supply an infinite density to trigger the non-finite safety audit.
    double inf_rho = __builtin_inf();

    EXPECT_THROW({
        get_rho_over_dt(dt, inf_rho);
    }, std::runtime_error);
}
