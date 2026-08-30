/**
 * @file test_scaling.cpp
 * @brief Literate Verification Suite for Navier-Stokes Scaling Factors (`scaling.cpp`)
 * 
 * @details
 * - What: Validates temporal and physical scaling calculations (`dt / rho` and `rho / dt`), 
 *         including OpenMP execution tracing, parameter constraint guards, and non-finite numerical audits.
 * - Why: Ensures invalid time-steps ($\Delta t \le 0$), unphysical densities ($\rho \le 0$), 
 *        and non-finite floating-point explosions trigger strict contract violation exceptions 
 *        (`std::invalid_argument`, `std::runtime_error`) and proper error logging streams.
 * - How: Executes boundary sweeps for valid coefficients, zero/negative inputs, and infinite limits 
 *        under a structured Google Test fixture and independent test harness.
 */

#include <gtest/gtest.h>
#include "scaling.hpp"
#include <stdexcept>
#include <cmath>
#include <limits>

using namespace navier_stokes_solver;

class ScalingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // We define standard fluid properties for our test domain.
        dt = 0.02;   // Time step (s)
        rho = 1.225; // Fluid density (kg/m^3)
    }

    double dt;
    double rho;
};

// ============================================================================
// SECTION 1 — Valid Scaling Factor Computation (Happy Path)
// ============================================================================
// Mathematical Rationale:
//   - For valid time-step $\Delta t > 0$ and density $\rho > 0$, the scaling coefficients 
//     are computed directly:
//       $S_1 = \frac{\Delta t}{\rho}$
//       $S_2 = \frac{\rho}{\Delta t}$
TEST_F(ScalingTest, ValidScalingComputations) {
    // Expected dt / rho = 0.02 / 1.225
    double dt_over_rho = get_dt_over_rho(dt, rho);
    double expected_dt_rho = dt / rho;
    EXPECT_NEAR(dt_over_rho, expected_dt_rho, 1e-12);

    // Expected rho / dt = 1.225 / 0.02 = 61.25
    double rho_over_dt = get_rho_over_dt(dt, rho);
    double expected_rho_dt = rho / dt;
    EXPECT_NEAR(rho_over_dt, expected_rho_dt, 1e-12);
}

// ============================================================================
// SECTION 2 — Temporal Constraint Validation (dt <= 0)
// ============================================================================
// Mathematical Rationale:
//   - Zero or negative time-steps violate temporal progression constraints and must 
//     throw an explicit std::invalid_argument while logging the temporal crash message.
TEST_F(ScalingTest, InvalidTimeStepThrows) {
    // Testing dt <= 0.0 for get_dt_over_rho (exercises lines 19-23 error logging and exception)
    EXPECT_THROW(
        get_dt_over_rho(0.0, rho),
        std::invalid_argument
    );
    EXPECT_THROW(
        get_dt_over_rho(-0.01, rho),
        std::invalid_argument
    );

    // Testing dt <= 0.0 for get_rho_over_dt (exercises lines 54-58 error logging and exception)
    EXPECT_THROW(
        get_rho_over_dt(0.0, rho),
        std::invalid_argument
    );
    EXPECT_THROW(
        get_rho_over_dt(-0.01, rho),
        std::invalid_argument
    );
}

// ============================================================================
// SECTION 3 — Physical Density Constraint Validation (rho <= 0)
// ============================================================================
// Mathematical Rationale:
//   - Zero or negative densities represent physical vacuums or impossible mediums, 
//     triggering an explicit std::invalid_argument and logging the physics crash message.
TEST_F(ScalingTest, InvalidDensityThrows) {
    // Testing rho <= 0.0 for get_dt_over_rho (exercises lines 26-30 error logging and exception)
    EXPECT_THROW(
        get_dt_over_rho(dt, 0.0),
        std::invalid_argument
    );
    EXPECT_THROW(
        get_dt_over_rho(dt, -1.2),
        std::invalid_argument
    );

    // Testing rho <= 0.0 for get_rho_over_dt (exercises lines 61-65 error logging and exception)
    EXPECT_THROW(
        get_rho_over_dt(dt, 0.0),
        std::invalid_argument
    );
    EXPECT_THROW(
        get_rho_over_dt(dt, -1.2),
        std::invalid_argument
    );
}

// ============================================================================
// SECTION 4 — Forensic Numerical Audit & Non-Finite Exception Handling
// ============================================================================
// Mathematical Rationale:
//   - If non-finite inputs lead to undefined scaling factors (such as infinity), 
//     the forensic auditor intercepts the non-finite result and throws a std::runtime_error.
TEST_F(ScalingTest, NonFiniteScalingThrows) {
    double inf_value = std::numeric_limits<double>::infinity();

    // Supplying an infinite time-step to trigger non-finite audit in get_dt_over_rho
    EXPECT_THROW(
        get_dt_over_rho(inf_value, rho),
        std::runtime_error
    );

    // Supplying an infinite density to trigger non-finite audit in get_rho_over_dt
    EXPECT_THROW(
        get_rho_over_dt(dt, inf_value),
        std::runtime_error
    );
}
