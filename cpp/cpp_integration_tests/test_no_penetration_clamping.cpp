/**
 * @file test_no_penetration_clamping.cpp
 * @brief Integration test verifying pressure-based velocity projection at fluid-solid interfaces during the corrector step.
 * 
 * ============================================================================
 * WHAT THIS TEST IS EVALUATING:
 * ============================================================================
 * This integration test validates that the pressure-velocity projection (corrector step) 
 * properly uses the pressure gradient to project trial velocities ($u^*$) near solid walls. 
 * Specifically, it tests whether a boundary-conforming pressure gradient correctly projects 
 * a trial velocity directed toward a solid wall down to zero, maintaining mathematical consistency 
 * with the Helmholtz-Hodge decomposition.
 * 
 * ============================================================================
 * WHY THIS TEST IS CRITICAL:
 * ============================================================================
 * Without proper pressure gradient coupling at fluid-solid boundaries, numerical methods 
 * can suffer from artificial wall penetration and momentum leakage. This test acts as a 
 * regression guard to ensure the pure MAC-grid projection operator correctly responds 
 * to boundary pressures.
 */

#include <gtest/gtest.h>
#include "corrector.hpp"
#include "grid_math.hpp"
#include <vector>

namespace navier_stokes_solver {

TEST(CorrectorIntegrationTest, NoPenetrationClampingAtSolidInterfaces) {
    // We define a small 3x3x3 Cartesian grid domain to isolate boundary interactions 
    // without global domain overhead.
    const int nx = 3;
    const int ny = 3;
    const int nz = 3;
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double rho = 1.0;
    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // Initialize velocity components and pressure fields to zero.
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1); // By default, all cells are active fluid cells (mask == 1).

    // Designate the center cell (1, 1, 1) as fluid, and the adjacent east cell (2, 1, 1) 
    // as a solid wall boundary (mask == 0).
    const size_t center_idx = static_cast<size_t>(get_flat_index(1, 1, 1, nx, ny));
    const size_t east_idx = static_cast<size_t>(get_flat_index(2, 1, 1, nx, ny));
    
    mask[east_idx] = 0; // Solid boundary on the east face.

    // We inject a trial velocity component pointing directly into the solid boundary:
    //     u_star = 2.0 m/s
    u_star[center_idx] = 2.0;

    // Set the pressure at the solid east boundary such that the pressure gradient 
    // analytically counteracts the trial velocity via pure projection:
    //     u_new = u_star - (dt / rho) * (p_east - p_center) / dx = 0.0
    //     0.0 = 2.0 - (0.01 / 1.0) * (p_east - 0.0) / 1.0  =>  p_east = 200.0
    p[east_idx] = 200.0;

    // We execute the pure corrector projection step:
    //     u_new = u_star - (dt / rho) * grad(p)
    solve_corrector_parallel(
        u, v, w,
        u_star, v_star, w_star,
        p, mask,
        nx, ny, nz,
        dx, dy, dz,
        dt, rho
    );

    // Verify that the pressure gradient successfully projects the velocity 
    // at the fluid-solid interface to zero:
    //     u_projected = 0.0 m/s
    ASSERT_NEAR(u[center_idx], 0.0, 1e-12);
}

} // namespace navier_stokes_solver