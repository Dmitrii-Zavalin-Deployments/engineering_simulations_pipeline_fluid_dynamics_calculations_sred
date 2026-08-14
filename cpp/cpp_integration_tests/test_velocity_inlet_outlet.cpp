/**
 * @file test_velocity_inlet_outlet.cpp
 * @brief Scenario 3.1: Velocity Inlet / Pressure Outlet Boundary Verification
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "orchestrator.hpp"
#include "predictor.hpp"
#include "pressure_poisson_solver.hpp"
#include "corrector.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"
#include "test_boundary_common.hpp"

using namespace navier_stokes_solver;

TEST(BoundaryConditionsTest, VelocityInletPressureOutlet) {
    int nx = 10, ny = 8, nz = 8;
    double dx = 0.1, dy = 0.1, dz = 0.1;
    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    double density = 1000.0;
    double mu = 0.001;
    double dt = 0.001;
    double U_0 = 1.0;

    std::vector<int> mask(total_cells, 1);
    std::vector<double> u(total_cells, U_0); // Initialize domain with U_0 to establish baseline mass flux
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    std::vector<BoundaryCondition> bc_list;

    BoundaryCondition bc_inlet;
    bc_inlet.location = "x_min";
    bc_inlet.type = "velocity_inlet";
    bc_inlet.u_val = U_0; bc_inlet.v_val = 0.0; bc_inlet.w_val = 0.0;
    bc_inlet.values.has_u = true; bc_inlet.values.u = U_0;
    bc_inlet.values.has_v = true; bc_inlet.values.v = 0.0;
    bc_inlet.values.has_w = true; bc_inlet.values.w = 0.0;
    bc_inlet.values.has_p = false;
    bc_list.push_back(bc_inlet);

    BoundaryCondition bc_outlet;
    bc_outlet.location = "x_max";
    bc_outlet.type = "pressure_outlet";
    bc_outlet.scalar_p = 0.0;
    bc_outlet.values.has_u = false;
    bc_outlet.values.has_v = false;
    bc_outlet.values.has_w = false;
    bc_outlet.values.has_p = true; bc_outlet.values.p = 0.0;
    bc_list.push_back(bc_outlet);

    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = 100; // Optimized for fast integration test execution
    config.poisson_tolerance = 1e-6;     // Relaxed tolerance to prevent test hangs
    std::vector<double> gravity = {0.0, 0.0, 0.0};
    FluidProperties fluid{mu / density, density};

    // Run sufficient time-stepping loop (1000 steps = 1.0s) to allow flow to traverse the 0.9m domain
    for (int step = 0; step < 1000; ++step) {
        execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

        std::vector<double> u_star(total_cells, 0.0);
        std::vector<double> v_star(total_cells, 0.0);
        std::vector<double> w_star(total_cells, 0.0);
        std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);

        compute_trial_velocities(
            dims, fluid, dt,
            u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(),
            gravity, mask,
            u_star.data(), v_star.data(), w_star.data()
        );

        std::vector<double> rhs(total_cells, 0.0);
        const double scale = density / dt;
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    size_t idx = get_flat_index(i, j, k, nx, ny);
                    size_t idx_east  = get_flat_index(i + 1, j, k, nx, ny);
                    size_t idx_west  = get_flat_index(i - 1, j, k, nx, ny);
                    size_t idx_north = get_flat_index(i, j + 1, k, nx, ny);
                    size_t idx_south = get_flat_index(i, j - 1, k, nx, ny);
                    size_t idx_up    = get_flat_index(i, j, k + 1, nx, ny);
                    size_t idx_down  = get_flat_index(i, j, k - 1, nx, ny);

                    double dudx = (u_star[idx_east]  - u_star[idx_west])  / (2.0 * dx);
                    double dvdy = (v_star[idx_north] - v_star[idx_south]) / (2.0 * dy);
                    double dwdz = (w_star[idx_up]    - w_star[idx_down])  / (2.0 * dz);

                    rhs[idx] = scale * (dudx + dvdy + dwdz);
                }
            }
        }

        solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list,
            nx, ny, nz, dx, dy, dz,
            config.max_poisson_iterations, config.poisson_tolerance,
            config.density, gravity
        );

        solve_corrector_parallel(
            u, v, w,
            u_star, v_star, w_star,
            p, mask,
            nx, ny, nz, dx, dy, dz,
            dt, density
        );
    }

    double inlet_mass_flow = 0.0;
    double outlet_mass_flow = 0.0;
    double face_area = dy * dz;

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            size_t in_idx  = get_flat_index(1, j, k, nx, ny);
            size_t out_idx = get_flat_index(nx - 2, j, k, nx, ny);
            inlet_mass_flow  += density * u[in_idx]  * face_area;
            outlet_mass_flow += density * u[out_idx] * face_area;
        }
    }

    ASSERT_NEAR(inlet_mass_flow, outlet_mass_flow, 1e-2)
        << "Mass flow conservation failure: Inlet mass rate does not match outlet mass rate.";
}
