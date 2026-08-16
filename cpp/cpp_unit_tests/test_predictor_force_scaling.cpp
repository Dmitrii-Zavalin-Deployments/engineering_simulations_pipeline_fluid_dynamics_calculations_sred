/**
 * @file test_predictor_force_scaling.cpp
 * @brief Literate Test Suite for Volumetric Body Force Density Scaling in Predictor Kernel
 *
 * This test file acts as a narrative document. Explanatory text and physical 
 * formulas are written as commented prose using ASCII formatting, while the executable C++ assertions 
 * verify that volumetric body forces are correctly scaled by fluid density during trial velocity updates.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "predictor.hpp"
#include "grid_math.hpp"

namespace navier_stokes_solver {

// ============================================================================
// NARRATIVE SECTION: Volumetric Force Density Scaling Verification
// ============================================================================
// In the momentum equation, external body forces are supplied as volumetric forces 
// with units of N/m^3. To compute the acceleration contribution, the volumetric force 
// must be divided by the fluid mass density (rho):
//     a_y = f_y / rho
// 
// The explicit Forward-Euler update for the trial velocity component then becomes:
//     v* = v_0 + dt * (f_y / rho)
// ============================================================================

TEST(PredictorForceScalingTest, VolumetricForceDensityScaling) {
    // 1. Grid Setup: Minimal 3x3x3 domain
    GridDimensions dims{3, 3, 3, 1.0, 1.0, 1.0};

    // 2. Simulation & Physical Properties: dt = 0.1 s, Water density (1000 kg/m^3), zero viscosity
    double dt = 0.1;
    FluidProperties fluid;
    fluid.nu = 0.0;             // Kinematic viscosity nu = 0 m^2/s
    fluid.density = 1000.0;     // Mass density rho = 1000 kg/m^3

    const size_t total_cells = static_cast<size_t>(dims.nx * dims.ny * dims.nz);

    // 3. Initial Quiescent Velocity Field (u = v = w = 0)
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);

    // 4. Volumetric Body Force Injection: 
    //     f_y = rho * g_y = 1000 kg/m^3 * (-9.81 m/s^2) = -9810 N/m^3
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, -9810.0);
    std::vector<double> fz(total_cells, 0.0);

    // 5. Gravity Vector & Active Fluid Mask
    std::vector<double> gravity(3, 0.0);
    std::vector<int> mask(total_cells, 1);

    // Output trial velocity buffers
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);

    // 6. Execute Predictor Kernel
    compute_trial_velocities(
        dims, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity,
        mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // 7. Dimensional Verification:
    // Physical acceleration:
    //     a_y = f_y / density = -9810.0 / 1000.0 = -9.81 m/s^2
    // Expected trial velocity:
    //     v* = v_0 + dt * a_y = 0.0 + (0.1 * -9.81) = -0.981 m/s
    const double expected_v_star = -0.981;

    // Verify interior cell (1, 1, 1) where stencil is active
    size_t idx = static_cast<size_t>(get_flat_index(1, 1, 1, dims.nx, dims.ny));
    EXPECT_NEAR(v_star[idx], expected_v_star, 1e-12)
        << "Dimensional Failure: Volumetric force f_y was not scaled by fluid.density at index " << idx;
    
    EXPECT_DOUBLE_EQ(u_star[idx], 0.0);
    EXPECT_DOUBLE_EQ(w_star[idx], 0.0);
}

} // namespace navier_stokes_solver
