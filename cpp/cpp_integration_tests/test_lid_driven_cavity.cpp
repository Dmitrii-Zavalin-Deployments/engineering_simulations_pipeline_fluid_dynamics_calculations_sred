/**
 * ============================================================================================
 * TEST SUITE: LidDrivenCavityTest
 * TEST CASE:  CavityFlowRe400
 * ============================================================================================
 *
 * WHAT THIS TEST VERIFIES:
 * 1. Hydrodynamic stability and primary recirculation vortex formation in a 3D enclosed 
 *    lid-driven cavity at Reynolds number Re = 400.
 * 2. Non-linear convective flux evaluations and high-shear momentum transport routines 
 *    within `cpp/src/predictor.cpp`.
 * 3. Boundary shear stress adjustments and velocity field updates in `cpp/src/corrector.cpp`.
 * 4. Convergence of the pressure Poisson solver (`cpp/src/pressure_poisson_solver.cpp`) 
 *    under enclosed wall-bounded boundary conditions with moving lid driving forces.
 *
 * HOW IT IS TESTED:
 * - A 3D computational domain of size L x H x W = 1.0 x 1.0 x 0.1 is discretized into a 
 *   32 x 32 x 3 staggered grid.
 * - Non-slip wall boundary conditions are applied on x_min, x_max, y_min, z_min, and z_max.
 * - A tangential moving lid boundary condition with velocity U_lid = 1.0 m/s is imposed 
 *   at the top boundary (y_max).
 * - The fluid dynamic viscosity is set to mu = 0.0025 Pa·s to maintain Re = 400 given 
 *   rho = 1.0 kg/m^3 and L = 1.0 m.
 * - The fractional-step solver advances the state over 100 time steps (dt = 0.001 s).
 * - In-flight assertions verify velocity boundedness and divergence limits using `compute_divergence`.
 * - Final physical verification checks that the top shear layer drives positive x-momentum 
 *   near the lid while generating a negative return-flow core at the cavity mid-height.
 *
 * WHY THIS IS CRITICAL FOR CODE COVERAGE:
 * - Tests non-zero tangential velocity wall boundary condition pathways across predictor 
 *   and corrector stages.
 * - Triggers high-gradient limiter and convection flux calculation branches in predictor.cpp.
 * - Verifies robust pressure Poisson matrix inversion under zero-net-flux Neumann constraints.
 * ============================================================================================
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#include "orchestrator.hpp"
#include "divergence.hpp"
#include "forces.hpp"

using namespace navier_stokes_solver;

TEST(LidDrivenCavityTest, CavityFlowRe400) {
    // We define a 3D rectangular domain Omega = [0, L] x [0, H] x [0, W]
    // discretized using a staggered finite-volume grid of dimensions 32 x 32 x 3 cells.
    const int nx = 32;
    const int ny = 32;
    const int nz = 3;

    const double L = 1.0;
    const double H = 1.0;
    const double W = 0.1;

    // The grid spacing in each spatial coordinate direction is calculated as:
    //     dx = L / nx,  dy = H / ny,  dz = W / nz
    const double dx = L / nx;
    const double dy = H / ny;
    const double dz = W / nz;
    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // We set the characteristic dimensionless parameter (Reynolds number):
    //     Re = (U_lid * L) / nu = 400
    // For a unit lid velocity U_lid = 1.0 m/s and length scale L = 1.0 m, 
    // the dynamic viscosity is computed directly as:
    //     mu = (U_lid * L) / Re = 1.0 / 400 = 0.0025 Pa·s
    const double u_lid = 1.0;
    const double Re = 400.0;
    const double mu = (u_lid * L) / Re;
    const double gravity[3] = {0.0, 0.0, 0.0};

    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1); // 1 indicates fluid domain

    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);

    // We construct the boundary condition list enforcing a moving lid on y_max and 
    // stationary no-slip conditions on all other domain boundaries.
    std::vector<BoundaryCondition> bc_list;

    // Top Boundary (y = H, "y_max"): Imposes moving lid velocity U = (1.0, 0.0, 0.0) m/s.
    BoundaryCondition bc_top;
    bc_top.location = "y_max";
    bc_top.type = "wall";
    bc_top.u_val = u_lid; bc_top.v_val = 0.0; bc_top.w_val = 0.0;
    bc_top.values.has_u = true; bc_top.values.u = u_lid;
    bc_top.values.has_v = true; bc_top.values.v = 0.0;
    bc_top.values.has_w = true; bc_top.values.w = 0.0;
    bc_list.push_back(bc_top);

    // Bottom Boundary (y = 0, "y_min"): Imposes zero velocity U = (0.0, 0.0, 0.0) m/s.
    BoundaryCondition bc_bottom;
    bc_bottom.location = "y_min";
    bc_bottom.type = "wall";
    bc_bottom.u_val = 0.0; bc_bottom.v_val = 0.0; bc_bottom.w_val = 0.0;
    bc_bottom.values.has_u = true; bc_bottom.values.u = 0.0;
    bc_bottom.values.has_v = true; bc_bottom.values.v = 0.0;
    bc_bottom.values.has_w = true; bc_bottom.values.w = 0.0;
    bc_list.push_back(bc_bottom);

    // Left Boundary (x = 0, "x_min"): Imposes zero velocity U = (0.0, 0.0, 0.0) m/s.
    BoundaryCondition bc_left;
    bc_left.location = "x_min";
    bc_left.type = "wall";
    bc_left.u_val = 0.0; bc_left.v_val = 0.0; bc_left.w_val = 0.0;
    bc_left.values.has_u = true; bc_left.values.u = 0.0;
    bc_left.values.has_v = true; bc_left.values.v = 0.0;
    bc_left.values.has_w = true; bc_left.values.w = 0.0;
    bc_list.push_back(bc_left);

    // Right Boundary (x = L, "x_max"): Imposes zero velocity U = (0.0, 0.0, 0.0) m/s.
    BoundaryCondition bc_right;
    bc_right.location = "x_max";
    bc_right.type = "wall";
    bc_right.u_val = 0.0; bc_right.v_val = 0.0; bc_right.w_val = 0.0;
    bc_right.values.has_u = true; bc_right.values.u = 0.0;
    bc_right.values.has_v = true; bc_right.values.v = 0.0;
    bc_right.values.has_w = true; bc_right.values.w = 0.0;
    bc_list.push_back(bc_right);

    // Front & Back Boundaries (z = 0, z = W): Impose zero velocity U = (0.0, 0.0, 0.0) m/s.
    for (const std::string& loc : {"z_min", "z_max"}) {
        BoundaryCondition bc_z;
        bc_z.location = loc;
        bc_z.type = "wall";
        bc_z.u_val = 0.0; bc_z.v_val = 0.0; bc_z.w_val = 0.0;
        bc_z.values.has_u = true; bc_z.values.u = 0.0;
        bc_z.values.has_v = true; bc_z.values.v = 0.0;
        bc_z.values.has_w = true; bc_z.values.w = 0.0;
        bc_list.push_back(bc_z);
    }

    // We instantiate the orchestrator solver with spatial discretization metadata
    // and configure fixed explicit time integration parameters dt = 0.001 s, max_steps = 100.
    Orchestrator orchestrator(nx, ny, nz, dx, dy, dz);
    const double dt = 0.001;
    const int max_steps = 100;

    std::vector<double> div_field(total_cells, 0.0);

    for (int step = 0; step < max_steps; ++step) {
        orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

        compute_divergence(u.data(), v.data(), w.data(), div_field.data(), nx, ny, nz, dx, dy, dz);
        double current_div = 0.0;
        for (size_t i = 0; i < total_cells; ++i) {
            ASSERT_TRUE(std::isfinite(div_field[i]));
            current_div = std::max(current_div, std::abs(div_field[i]));
        }

        ASSERT_TRUE(std::isfinite(current_div));
        ASSERT_LT(current_div, 5.0);

        // We verify numerical stability by asserting that maximum field velocities do not exceed
        // 1.5 times the driving lid speed (U_max <= 1.5 * U_lid).
        double max_u = 0.0;
        for (size_t i = 0; i < total_cells; ++i) {
            ASSERT_TRUE(std::isfinite(u[i]));
            ASSERT_TRUE(std::isfinite(v[i]));
            max_u = std::max(max_u, std::abs(u[i]));
        }
        ASSERT_LE(max_u, u_lid * 1.5);
    }

    // We validate the physical structure of the developing primary recirculation vortex.
    // In lid-driven cavity flow:
    //     - Upper fluid cells near the top lid (y -> H) acquire momentum in the positive x-direction (u > 0).
    //     - Lower fluid cells along the vertical mid-plane (y ~ H/4) form the recirculation return flow (u < 0).
    const int k_plane = 1;
    const int mid_x = nx / 2;

    // We check the upper cavity region (y_index = ny - 3):
    // Expected physical state: u_computed > 0.0 m/s
    const size_t idx_upper = mid_x + static_cast<size_t>(nx) * ((ny - 3) + ny * k_plane);
    EXPECT_GT(u[idx_upper], 0.0);

    // We check the lower return flow region (y_index = ny / 4):
    // Expected physical state: u_computed < 0.0 m/s
    const size_t idx_lower = mid_x + static_cast<size_t>(nx) * (ny / 4 + ny * k_plane);
    EXPECT_LT(u[idx_lower], 0.0);
}
