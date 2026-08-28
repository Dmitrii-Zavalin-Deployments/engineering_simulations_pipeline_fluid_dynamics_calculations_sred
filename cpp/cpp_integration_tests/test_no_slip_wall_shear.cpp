/**
 * @file test_no_slip_wall_shear.cpp
 * @brief Scenario 3.3: No-Slip Wall Shear Boundary Verification
 *
 * LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
 * ---------------------------------------------------------------------------------
 * No-slip wall boundaries enforce absolute velocity stagnation at solid boundaries:
 * 
 *     u_wall = 0.0, v_wall = 0.0, w_wall = 0.0
 * 
 * Furthermore, viscous shear forces create a boundary layer velocity profile where 
 * fluid velocity monotonically decreases from the core flow towards the stationary wall:
 * 
 *     u_near_wall < u_core
 * 
 * TEST SCENARIO:
 *   - We initialize the domain with a uniform velocity field (u = 1.0 m/s) and apply
 *     a no-slip wall condition at the y_min boundary (mask = -1).
 *   - We verify that the pre-step enforces zero velocity on the wall boundary faces.
 *   - We compute trial velocities via viscous diffusion and check that the near-wall 
 *     velocity decelerates relative to the core flow due to viscous shear.
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "orchestrator.hpp"
#include "predictor.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

TEST(BoundaryConditionsTest, NoSlipWallShearBoundary) {
    // We define the spatial grid dimensions and resolution:
    //     nx = 6, ny = 8, nz = 6
    //     dx = 0.1 m, dy = 0.1 m, dz = 0.1 m
    int nx = 6, ny = 8, nz = 6;
    double dx = 0.1, dy = 0.1, dz = 0.1;
    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // We define physical properties and time-step parameters:
    //     density = 1000.0 kg/m^3, mu = 0.1 Pa*s, dt = 0.001 s
    double density = 1000.0;
    double mu = 0.1;
    double dt = 0.001;

    // We initialize simulation fields with a uniform core velocity:
    //     u = 1.0 m/s, v = 0.0 m/s, w = 0.0 m/s, p = 0.0 Pa, mask = 1 (active fluid)
    std::vector<int> mask(total_cells, 1);
    std::vector<double> u(total_cells, 1.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // We carve out the bottom boundary wall at y_min (j = 0) with solid mask (mask = -1) and zero velocity:
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            size_t idx = get_flat_index(i, 0, k, nx, ny);
            mask[idx] = -1;
            u[idx] = 0.0;
        }
    }

    // We configure the no-slip boundary condition descriptor for the y_min wall:
    std::vector<BoundaryCondition> bc_list;
    BoundaryCondition bc_wall;
    bc_wall.location = "y_min";
    bc_wall.type = "no_slip_wall";
    bc_wall.u_val = 0.0; bc_wall.v_val = 0.0; bc_wall.w_val = 0.0;
    bc_wall.values.has_u = true; bc_wall.values.u = 0.0;
    bc_wall.values.has_v = true; bc_wall.values.v = 0.0;
    bc_wall.values.has_w = true; bc_wall.values.w = 0.0;
    bc_wall.values.has_p = false;
    bc_list.push_back(bc_wall);

    // We execute the pre-step to enforce boundary conditions across the grid:
    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz, false);

    // We verify that no-slip conditions successfully maintain zero velocity directly on the wall face:
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            size_t idx = get_flat_index(i, 0, k, nx, ny);
            ASSERT_DOUBLE_EQ(u[idx], 0.0) 
                << "No-slip violation: Non-zero velocity detected directly on wall boundary face.";
        }
    }

    // We allocate trial velocity storage and fluid property structures for viscous diffusion:
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);
    std::vector<double> gravity = {0.0, 0.0, 0.0};
    FluidProperties fluid{mu / density, density};

    // We compute trial velocities incorporating viscous shear diffusion:
    compute_trial_velocities(
        dims, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity, mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // We evaluate near-wall deceleration relative to the core flow:
    size_t near_wall_idx = get_flat_index(2, 1, 2, nx, ny);
    size_t core_idx      = get_flat_index(2, 5, 2, nx, ny);

    ASSERT_LT(u_star[near_wall_idx], u_star[core_idx])
        << "Viscous boundary layer failure: Near-wall velocity did not decelerate relative to core flow.";
}
