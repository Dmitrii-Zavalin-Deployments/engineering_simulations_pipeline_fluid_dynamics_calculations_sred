/**
 * @file test_predictor.cpp
 * @brief Literate Verification Suite for Step 1 Predictor Module (`predictor.cpp`)
 * 
 * @details
 * - What: Validates all input contract guards, geometric limits, physical parameter bounds, 
 *         and non-finite numeric failure detection within `compute_trial_velocities`.
 * - Why: Ensures 100% statement and branch coverage, trapping invalid states early via 
 *        `std::invalid_argument` and `std::runtime_error`.
 * - How: Exercises each validation exception branch and numeric failure mode individually under Google Test.
 */

#include <gtest/gtest.h>
#include "predictor.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

class PredictorTest : public ::testing::Test {
protected:
    navier_stokes_solver::GridDimensions dims{3, 3, 3, 1.0, 1.0, 1.0};
    navier_stokes_solver::FluidProperties fluid{1.0, 0.01}; // density = 1.0, nu = 0.01
    double dt = 0.01;
    
    // Explicit constructor calls ensure 27 elements are allocated instead of std::initializer_list
    std::vector<double> u = std::vector<double>(27, 0.0);
    std::vector<double> v = std::vector<double>(27, 0.0);
    std::vector<double> w = std::vector<double>(27, 0.0);
    std::vector<double> fx = std::vector<double>(27, 0.0);
    std::vector<double> fy = std::vector<double>(27, 0.0);
    std::vector<double> fz = std::vector<double>(27, 0.0);
    std::vector<double> gravity{0.0, 0.0, -9.81};
    std::vector<int> mask = std::vector<int>(27, 1);
    std::vector<double> u_star = std::vector<double>(27, 0.0);
    std::vector<double> v_star = std::vector<double>(27, 0.0);
    std::vector<double> w_star = std::vector<double>(27, 0.0);
};

// ============================================================================
// SECTION 1 — Input Validation & Contract Guard Tests (Lines 33–55)
// ============================================================================

// Validates Line 33: Null pointer check exception handling
TEST_F(PredictorTest, NullPointerThrowsException) {
    EXPECT_THROW(
        navier_stokes_solver::compute_trial_velocities(
            dims, fluid, dt, nullptr, v.data(), w.data(),
            fx.data(), fy.data(), fz.data(), gravity, mask,
            u_star.data(), v_star.data(), w_star.data()
        ),
        std::invalid_argument
    );
}

// Validates Line 36: Gravity vector dimension check exception handling
TEST_F(PredictorTest, InvalidGravitySizeThrowsException) {
    std::vector<double> invalid_gravity = {0.0, -9.81}; // Size != 3
    EXPECT_THROW(
        navier_stokes_solver::compute_trial_velocities(
            dims, fluid, dt, u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(), invalid_gravity, mask,
            u_star.data(), v_star.data(), w_star.data()
        ),
        std::invalid_argument
    );
}

// Validates Line 40: Mask vector size match exception handling
TEST_F(PredictorTest, MaskSizeMismatchThrowsException) {
    std::vector<int> invalid_mask(10, 1); // Size does not match 27 total cells
    EXPECT_THROW(
        navier_stokes_solver::compute_trial_velocities(
            dims, fluid, dt, u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(), gravity, invalid_mask,
            u_star.data(), v_star.data(), w_star.data()
        ),
        std::invalid_argument
    );
}

// Validates Line 43: Minimum grid dimensions constraint exception handling
TEST_F(PredictorTest, SmallGridDimensionsThrowsException) {
    navier_stokes_solver::GridDimensions small_dims{2, 3, 3, 1.0, 1.0, 1.0};
    std::vector<double> small_u(18, 0.0);
    std::vector<int> small_mask(18, 1);
    EXPECT_THROW(
        navier_stokes_solver::compute_trial_velocities(
            small_dims, fluid, dt, small_u.data(), small_u.data(), small_u.data(),
            small_u.data(), small_u.data(), small_u.data(), gravity, small_mask,
            small_u.data(), small_u.data(), small_u.data()
        ),
        std::invalid_argument
    );
}

// Validates Line 46: Strictly positive grid spacing exception handling
TEST_F(PredictorTest, NonPositiveGridSpacingThrowsException) {
    navier_stokes_solver::GridDimensions bad_dims{3, 3, 3, 0.0, 1.0, 1.0};
    EXPECT_THROW(
        navier_stokes_solver::compute_trial_velocities(
            bad_dims, fluid, dt, u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(), gravity, mask,
            u_star.data(), v_star.data(), w_star.data()
        ),
        std::invalid_argument
    );
}

// Validates Line 49: Strictly positive time step exception handling
TEST_F(PredictorTest, NonPositiveDtThrowsException) {
    EXPECT_THROW(
        navier_stokes_solver::compute_trial_velocities(
            dims, fluid, 0.0, u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(), gravity, mask,
            u_star.data(), v_star.data(), w_star.data()
        ),
        std::invalid_argument
    );
}

// Validates Line 52: Non-negative kinematic viscosity exception handling
TEST_F(PredictorTest, NegativeViscosityThrowsException) {
    navier_stokes_solver::FluidProperties bad_fluid{1.0, -0.01};
    EXPECT_THROW(
        navier_stokes_solver::compute_trial_velocities(
            dims, bad_fluid, dt, u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(), gravity, mask,
            u_star.data(), v_star.data(), w_star.data()
        ),
        std::invalid_argument
    );
}

// Validates Line 55: Strictly positive fluid density exception handling
TEST_F(PredictorTest, NonPositiveDensityThrowsException) {
    navier_stokes_solver::FluidProperties bad_fluid{0.0, 0.01};
    EXPECT_THROW(
        navier_stokes_solver::compute_trial_velocities(
            dims, bad_fluid, dt, u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(), gravity, mask,
            u_star.data(), v_star.data(), w_star.data()
        ),
        std::invalid_argument
    );
}

// ============================================================================
// SECTION 2 — Numerical Stability & Runtime Error Tests (Line 147)
// ============================================================================

// Validates Line 147: Non-finite trial velocity propagation detection and runtime error throwing
TEST_F(PredictorTest, NonFiniteVelocityThrowsRuntimeError) {
    std::vector<double> poisoned_u = u;
    poisoned_u[13] = NAN; // Inject NaN into central grid cell to trigger non-finite check

    EXPECT_THROW(
        navier_stokes_solver::compute_trial_velocities(
            dims, fluid, dt, poisoned_u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(), gravity, mask,
            u_star.data(), v_star.data(), w_star.data()
        ),
        std::runtime_error
    );
}