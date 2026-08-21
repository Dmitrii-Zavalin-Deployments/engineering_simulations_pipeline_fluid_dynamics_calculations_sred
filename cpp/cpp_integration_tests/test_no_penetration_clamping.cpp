/**
 * @file test_no_penetration_clamping.cpp
 * @brief Integration test verifying no-penetration velocity clamping at fluid-solid interfaces during corrector projection.
 */

#include <gtest/gtest.h>
#include "corrector.hpp"
#include "grid_math.hpp"
#include <vector>

namespace navier_stokes_solver {
namespace testing {

TEST(CorrectorIntegrationTest, NoPenetrationClampingAtSolidInterfaces) {
    const int nx = 3;
    const int ny = 3;
    const int nz = 3;
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double rho = 1.0;
    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1); // Fluid cells active by default

    // Setup center cell (1, 1, 1) as fluid, and east cell (2, 1, 1) as solid (mask == 0)
    const size_t center_idx = static_cast<size_t>(get_flat_index(1, 1, 1, nx, ny));
    const size_t east_idx = static_cast<size_t>(get_flat_index(2, 1, 1, nx, ny));
    
    mask[east_idx] = 0; // Solid boundary on the east

    // Provide a trial velocity pointing into the solid wall (positive u_star)
    u_star[center_idx] = 2.0;

    // Run the corrector projection step
    solve_corrector_parallel(
        u, v, w,
        u_star, v_star, w_star,
        p, mask,
        nx, ny, nz,
        dx, dy, dz,
        dt, rho
    );

    // Assert that the velocity component normal to the solid boundary is clamped to zero,
    // preventing artificial wall penetration or transverse spikes.
    EXPECT_NEAR(u[center_idx], 0.0, 1e-12);
}

} // namespace testing
} // namespace navier_stokes_solver
