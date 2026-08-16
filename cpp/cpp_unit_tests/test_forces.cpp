/**
 * @file test_forces.cpp
 * @brief Literate test suite for body force vector validation and numerical auditing.
 * 
 * LITERATE TESTING NARRATIVE & MATHEMATICAL FORMULATION:
 * ---------------------------------------------------------------------------------
 * In computational fluid dynamics, body forces like gravity or electromagnetic 
 * fields must be strictly three-dimensional (Fx, Fy, Fz) and mathematically 
 * finite to prevent simulation corruption or floating-point exceptions.
 * 
 * Body force vector representation:
 *     F = [Fx, Fy, Fz]^T
 * 
 * Dimension constraints:
 *     dim(F) == 3
 * 
 * Numerical finiteness constraint:
 *     isfinite(Fx) && isfinite(Fy) && isfinite(Fz) == true
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <array>
#include <limits>
#include <stdexcept>
#include <cassert>
#include "forces.hpp"

using namespace navier_stokes_solver;

/**
 * Test Case 1: Valid standard 3D force vector extraction
 * 
 * We define a standard 3-component body force vector (e.g., gravity and acceleration):
 *     Fx = 0.0
 *     Fy = -9.81
 *     Fz = 1.25
 */
TEST(ForcesTest, ValidForceVectorExtraction) {
    // We define a standard 3-component body force vector (e.g., gravity and acceleration):
    //     Fx = 0.0, Fy = -9.81, Fz = 1.25
    std::vector<double> valid_forces = {0.0, -9.81, 1.25};

    assert(valid_forces.size() == 3);

    // We validate and extract the force components into a fixed array.
    std::array<double, 3> extracted = validate_and_get_forces(valid_forces);

    // We verify that each component matches the expected physical value precisely:
    assert(std::abs(extracted[0] - 0.0) < 1e-9);
    assert(std::abs(extracted[1] - (-9.81)) < 1e-9);
    assert(std::abs(extracted[2] - 1.25) < 1e-9);

    EXPECT_DOUBLE_EQ(extracted[0], 0.0);
    EXPECT_DOUBLE_EQ(extracted[1], -9.81);
    EXPECT_DOUBLE_EQ(extracted[2], 1.25);
}

/**
 * Test Case 2: Invalid dimension guard (contract violation)
 * 
 * A vector containing fewer or more than 3 components violates the 3D spatial contract:
 *     dim(F) != 3 -> throw std::invalid_argument
 */
TEST(ForcesTest, InvalidVectorSizeThrows) {
    // A vector containing only 2 components violates the 3D spatial contract:
    std::vector<double> short_forces = {1.0, 2.0};

    // We expect the validator to throw a contract violation exception:
    EXPECT_THROW(validate_and_get_forces(short_forces), std::invalid_argument);

    // Similarly, a vector with 4 components is over-specified and invalid:
    std::vector<double> long_forces = {1.0, 2.0, 3.0, 4.0};
    EXPECT_THROW(validate_and_get_forces(long_forces), std::invalid_argument);
}

/**
 * Test Case 3: Forensic numerical audit (non-finite value protection)
 * 
 * If a NaN (Not a Number) or Infinity enters the force vector due to a math breakdown, 
 * the forensic auditor must intercept it immediately and throw a std::runtime_error:
 *     !isfinite(F_component) -> throw std::runtime_error
 */
TEST(ForcesTest, NonFiniteValuesThrow) {
    // If a NaN (Not a Number) enters the force vector due to a math breakdown, 
    // the forensic auditor must intercept it immediately:
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> corrupted_forces = {0.0, nan_val, 9.81};

    // We expect a runtime error to be thrown:
    EXPECT_THROW(validate_and_get_forces(corrupted_forces), std::runtime_error);

    // Testing with infinite values as well:
    std::vector<double> infinite_forces = {std::numeric_limits<double>::infinity(), 0.0, 0.0};
    EXPECT_THROW(validate_and_get_forces(infinite_forces), std::runtime_error);
}
