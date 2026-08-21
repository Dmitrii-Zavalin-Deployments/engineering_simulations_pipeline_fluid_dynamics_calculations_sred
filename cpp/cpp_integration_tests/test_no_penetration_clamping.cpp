/**
 * @file test_no_penetration_clamping.cpp
 * @brief Integration test verifying no-penetration velocity clamping at fluid-solid interfaces during corrector projection.
 * 
 * ============================================================================
 * WHAT THIS TEST IS EVALUATING:
 * ============================================================================
 * This integration test validates that the pressure-velocity projection (corrector step) 
 * properly enforces the physical no-penetration boundary condition at fluid-solid interfaces. 
 * Specifically, it tests whether trial velocity components ($u^*$) directed into a solid wall 
 * ($mask == 0$) are correctly projected and clamped to zero.
 * 
 * ============================================================================
 * WHY THIS TEST IS CRITICAL:
 * ============================================================================
 * Without explicit no-penetration boundary clamping during projection, unphysical pressure 
 * gradients at fluid-solid boundaries can force fluid velocities through solid walls. This 
 * leads to artificial wall penetration, momentum leakage, severe numerical instability, 
 * and pressure blow-ups. This test acts as a regression guard to ensure solid boundaries 
 * remain impenetrable.
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

    // We execute the corrector projection step to enforce mass conservation 
    // and project out non-divergent velocities:
    //     u_new = u_star - (dt / rho) * grad(p)
    solve_corrector_parallel(
        u, v, w,
        u_star, v_star, w_star,
        p, mask,
        nx, ny, nz,
        dx, dy, dz,
        dt, rho
    );

    // Under the no-penetration boundary condition doctrine, the velocity component 
    // normal to a solid wall interface must be strictly clamped to zero:
    //     u_projected = 0.0 m/s
    // We use a fatal assertion (ASSERT_NEAR) to ensure execution halts immediately 
    // if artificial wall penetration occurs.
    ASSERT_NEAR(u[center_idx], 0.0, 1e-12);
}

} // namespace navier_stokes_solver