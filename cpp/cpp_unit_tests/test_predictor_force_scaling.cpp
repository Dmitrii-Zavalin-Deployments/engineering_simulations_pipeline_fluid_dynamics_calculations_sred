/**
 * @file test_predictor_force_scaling.cpp
 * @brief Unit test verifying density scaling of volumetric body forces in predictor.cpp
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "predictor.hpp"
#include "grid_math.hpp"

namespace navier_stokes_solver {

TEST(PredictorForceScalingTest, VolumetricForceDensityScaling) {
    // 1. Grid Setup: Minimal 3x3x3 domain
    GridDimensions dims{3, 3, 3, 1.0, 1.0, 1.0};

    // 2. Simulation & Physical Properties: dt = 0.1 s, Water density (1000 kg/m^3), zero viscosity
    double dt = 0.1;
    FluidProperties fluid;
    fluid.nu = 0.0;           // Kinematic viscosity nu = 0 m^2/s
    fluid.density = 1000.0;   // Mass density rho = 1000 kg/m^3

    const size_t total_cells = static_cast<size_t>(dims.nx * dims.ny * dims.nz);

    // 3. Initial Quiescent Velocity Field (u = v = w = 0)
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);

    // 4. Volumetric Body Force Injection: f_y = rho * g_y = 1000 kg/m^3 * -9.81 m/s^2 = -9810 N/m^3
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, -9810.0);
    std::vector<double> fz(total_cells, 0.0);

    // 5. Active Fluid Mask (mask == 1)
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
        mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // 7. Dimensional Verification:
    // Physical acceleration a_y = f_y / density = -9810.0 / 1000.0 = -9.81 m/s^2
    // Expected v_star = v_0 + dt * a_y = 0.0 + (0.1 s * -9.81 m/s^2) = -0.981 m/s
    const double expected_v_star = -0.981;

    for (size_t idx = 0; idx < total_cells; ++idx) {
        EXPECT_NEAR(v_star[idx], expected_v_star, 1e-12)
            << "Dimensional Failure: Volumetric force f_y was not scaled by fluid.density at index " << idx;
        
        EXPECT_DOUBLE_EQ(u_star[idx], 0.0);
        EXPECT_DOUBLE_EQ(w_star[idx], 0.0);
    }
}

} // namespace navier_stokes_solver
