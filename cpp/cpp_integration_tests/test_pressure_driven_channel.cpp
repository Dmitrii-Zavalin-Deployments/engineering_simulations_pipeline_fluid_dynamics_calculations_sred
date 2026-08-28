/**
 * @file test_pressure_driven_channel.cpp
 * @brief Scenario 3.2: Pressure-Driven Channel Flow Verification
 *
 * LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
 * ---------------------------------------------------------------------------------
 * Pressure-driven channel flows are accelerated by imposing a pressure gradient 
 * along the streamwise direction (x-axis):
 * 
 *     dp/dx = (p_out - p_in) / L
 * 
 * According to the Navier-Stokes momentum equations, this pressure gradient acts as 
 * a driving body force that accelerates fluid from regions of high pressure to low pressure:
 * 
 *     du/dt = -(1 / rho) * (dp/dx) + nu * laplacian(u)
 * 
 * TEST SCENARIO:
 *   - We initialize a linear pressure drop from p_in = 10.0 Pa to p_out = 0.0 Pa across the channel.
 *   - We execute a multi-step time integration loop combining pre-step enforcement, trial velocity
 *     computation, pressure Poisson equation solving, and velocity correction.
 *   - We verify that the mean streamwise velocity in the channel core becomes positive (u > 0.0),
 *     confirming that the pressure gradient successfully accelerates the fluid from rest.
 * ---------------------------------------------------------------------------------
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

using namespace navier_stokes_solver;

TEST(BoundaryConditionsTest, PressureDrivenChannelFlow) {
    // We define grid dimensions and spatial step sizes:
    //     nx = 10, ny = 6, nz = 6
    //     dx = 0.1 m, dy = 0.1 m, dz = 0.1 m
    int nx = 10, ny = 6, nz = 6;
    double dx = 0.1, dy = 0.1, dz = 0.1;
    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // We define physical properties and boundary pressures:
    //     density = 1000.0 kg/m^3, mu = 0.001 Pa*s, dt = 0.001 s
    //     p_in = 10.0 Pa, p_out = 0.0 Pa
    double density = 1000.0;
    double mu = 0.001;
    double dt = 0.001;
    double p_in = 10.0;
    double p_out = 0.0;

    // We initialize velocity fields to rest (u = v = w = 0.0):
    std::vector<int> mask(total_cells, 1);
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    
    // We initialize pressure with a linear gradient from inlet to outlet to drive initial momentum:
    //     p(x) = p_in + (x / L) * (p_out - p_in)
    std::vector<double> p(total_cells, 0.0);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                double x_frac = (nx > 1) ? static_cast<double>(i) / (nx - 1) : 0.0;
                p[idx] = p_in + x_frac * (p_out - p_in);
            }
        }
    }

    // We configure boundary condition descriptors for pressure inlet and outlet:
    std::vector<BoundaryCondition> bc_list;

    BoundaryCondition bc_in;
    bc_in.location = "x_min";
    bc_in.type = "pressure_inlet";
    bc_in.scalar_p = p_in;
    bc_in.values.has_u = false;
    bc_in.values.has_v = false;
    bc_in.values.has_w = false;
    bc_in.values.has_p = true; bc_in.values.p = p_in;
    bc_list.push_back(bc_in);

    BoundaryCondition bc_out;
    bc_out.location = "x_max";
    bc_out.type = "pressure_outlet";
    bc_out.scalar_p = p_out;
    bc_out.values.has_u = false;
    bc_out.values.has_v = false;
    bc_out.values.has_w = false;
    bc_out.values.has_p = true; bc_out.values.p = p_out;
    bc_list.push_back(bc_out);

    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = 500;
    config.poisson_tolerance = 1e-12;
    std::vector<double> gravity = {0.0, 0.0, 0.0};
    FluidProperties fluid{mu / density, density};

    // We run a multi-step time integration loop to allow pressure gradient acceleration from rest:
    for (int step = 0; step < 150; ++step) {
        execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz, false);

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

    // We compute the mean streamwise velocity across the channel core region:
    double center_u_sum = 0.0;
    int count = 0;
    for (int k = 2; k <= 3; ++k) {
        for (int j = 2; j <= 3; ++j) {
            for (int i = 2; i <= 7; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                center_u_sum += u[idx];
                count++;
            }
        }
    }
    double mean_u = center_u_sum / count;

    ASSERT_GT(mean_u, 0.0) 
        << "Pressure-driven channel failed to accelerate fluid from high to low pressure.";
}
