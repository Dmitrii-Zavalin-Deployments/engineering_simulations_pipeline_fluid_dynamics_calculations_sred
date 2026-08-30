/**
 * @file test_predictor_coverage.cpp
 * @brief Literate verification suite for Navier-Stokes predictor boundary and stability guards.
 * 
 * WHAT: This test file verifies that the predictive velocity step and its input validation layer 
 *       correctly intercept unphysical parameters and numerical anomalies.
 * HOW: It establishes narrative verification steps combining mathematical expectations with 
 *      strict C++ GoogleTest assertions.
 * WHY: Achieving complete coverage of defensive checks ensures robustness against corrupted states, 
 *      divergent advection terms, or invalid fluid property declarations during high-performance simulation runs.
 */

#include <gtest/gtest.h>
#include "predictor.hpp"
#include "grid_math.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>

namespace navier_stokes_solver {
    // Forward declare validate_inputs to access internal contract validation logic directly
    void validate_inputs(
        const GridDimensions& dims,
        const FluidProperties& fluid,
        double dt,
        const double* u, const double* v, const double* w,
        const double* fx, const double* fy, const double* fz,
        const std::vector<double>& gravity,
        const std::vector<int>& mask,
        const double* u_star, const double* v_star, const double* w_star
    );
}

// We define a test case to ensure that unphysical fluid properties, specifically 
// a negative kinematic viscosity, are intercepted before any momentum calculation begins.
TEST(PredictorCoverageTest, NegativeViscosityThrows) {
    // The computational grid is initialized with standard dimensions for a 3x3x3 cell layout.
    navier_stokes_solver::GridDimensions dims{3, 3, 3, 0.1, 0.1, 0.1};
    
    // Kinematic viscosity nu must be non-negative (nu >= 0). 
    // Passing a negative value violates core fluid dynamics principles:
    //     nu < 0 => Invalid physics state
    navier_stokes_solver::FluidProperties fluid{-1.0, 1.0}; 
    double dt = 0.01;

    // We allocate flat velocity and force arrays matching the 27-cell grid requirement.
    const size_t total_cells = 27;
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);
    std::vector<double> gravity = {0.0, 0.0, -9.81};
    std::vector<int> mask(total_cells, 1);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);

    // The validation function must intercept the negative viscosity and throw std::invalid_argument.
    EXPECT_THROW(
        navier_stokes_solver::validate_inputs(
            dims, fluid, dt,
            u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(),
            gravity, mask,
            u_star.data(), v_star.data(), w_star.data()
        ),
        std::invalid_argument
    );
}

// We define a test case to catch numerical instabilities and floating-point explosions 
// during the forward-Euler temporal integration step.
TEST(PredictorCoverageTest, NonFiniteVelocityThrows) {
    navier_stokes_solver::GridDimensions dims{3, 3, 3, 0.1, 0.1, 0.1};
    navier_stokes_solver::FluidProperties fluid{1.0e-3, 1.0};
    double dt = 0.01;

    const size_t total_cells = 27;
    std::vector<double> u(total_cells, 0.0);
    
    // If an advection or diffusion term diverges to NaN at cell index 13 (center of the 3x3x3 domain),
    // the parallel reduction flag `has_non_finite` will capture it:
    //     u_t = u + dt * (-adv + nu * lap + f/rho + g) == NaN => Trigger Panic
    u[13] = NAN; 
    
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);
    std::vector<double> gravity = {0.0, 0.0, 0.0};
    std::vector<int> mask(total_cells, 1);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);

    // Computing trial velocities with a NaN-poisoned field must trigger a runtime panic.
    EXPECT_THROW(
        navier_stokes_solver::compute_trial_velocities(
            dims, fluid, dt,
            u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(),
            gravity, mask,
            u_star.data(), v_star.data(), w_star.data()
        ),
        std::runtime_error
    );
}