/**
 * @file test_waterfall_inflow.cpp
 * @brief Integration Test Suite for Waterfall Inflow Dynamics 
 *        and Gravity-Driven Pouring Flow.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include "orchestrator.hpp"
#include "predictor.hpp"
#include "pressure_poisson_solver.hpp"
#include "corrector.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"

using namespace navier_stokes_solver;

TEST(WaterfallDiagnosticTest, WaterfallInflowDynamics) {
    int nx = 9;
    int ny = 11; // Taller vertical domain to allow falling stream
    int nz = 9;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    double density = 1000.0;
    double mu = 0.001;
    double dt = 0.001;

    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = 500;
    config.poisson_tolerance = 1e-12;

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // Mask: Side and bottom walls, open top boundary container for waterfall
    std::vector<int> mask(total_cells, 1);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                if (i == 0 || i == nx - 1 || j == 0 || k == 0 || k == nz - 1) {
                    mask[idx] = -1;
                }
            }
        }
    }

    double gravity_y = -9.81;
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> rhs(total_cells, 0.0);

    std::vector<BoundaryCondition> bc_list;
    BoundaryCondition bc;
    bc.location = "wall";
    bc.type = "velocity";
    bc.u_val = 0.0; bc.v_val = 0.0; bc.w_val = 0.0; bc.scalar_p = 0.0;
    bc_list.push_back(bc);

    std::vector<double> gravity = {0.0, gravity_y, 0.0};
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);

    // 1. Execute Pre-Step Boundary Conditions FIRST
    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

    // 2. Initialize inflow stream AFTER pre-step to prevent boundary overwrite
    int top_j = ny - 1;
    for (int k = 3; k <= 5; ++k) {
        for (int i = 3; i <= 5; ++i) {
            size_t idx = get_flat_index(i, top_j, k, nx, ny);
            v[idx] = -0.5; // Downward injection velocity
        }
    }

    // Execute Predictor Step (Gravity accelerates inflow downward)
    FluidProperties fluid{mu / density, density};
    compute_trial_velocities(
        dims, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity, mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // Pressure Poisson RHS Divergence & Iterative Solve
    const double scale = density / dt;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t idx = get_flat_index(i, j, k, nx, ny);
                if (mask[idx] == 1) {
                    const size_t idx_east  = get_flat_index(i + 1, j, k, nx, ny);
                    const size_t idx_west  = get_flat_index(i - 1, j, k, nx, ny);
                    const size_t idx_north = get_flat_index(i, j + 1, k, nx, ny);
                    const size_t idx_south = get_flat_index(i, j - 1, k, nx, ny);
                    const size_t idx_up    = get_flat_index(i, j, k + 1, nx, ny);
                    const size_t idx_down  = get_flat_index(i, j, k - 1, nx, ny);

                    double dudx = (u_star[idx_east]  - u_star[idx_west])  / (2.0 * dx);
                    double dvdy = (v_star[idx_north] - v_star[idx_south]) / (2.0 * dy);
                    double dwdz = (w_star[idx_up]    - w_star[idx_down])  / (2.0 * dz);

                    rhs[idx] = scale * (dudx + dvdy + dwdz);
                } else {
                    rhs[idx] = 0.0;
                }
            }
        }
    }

    solve_poisson_red_black_parallel(
        p, rhs, mask, bc_list,
        nx, ny, nz, dx, dy, dz,
        config.max_poisson_iterations, config.poisson_tolerance,
        config.density, gravity
    );

    // Corrector Step Projection & Final Verification
    solve_corrector_parallel(
        u, v, w,
        u_star, v_star, w_star,
        p, mask,
        nx, ny, nz, dx, dy, dz,
        dt, density
    );

    double min_v_downstream = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (mask[idx] == 1) {
            min_v_downstream = std::min(min_v_downstream, v[idx]);
        }
    }

    // Assertion: Gravity and inflow must drive a negative (downward) stream velocity
    EXPECT_LT(min_v_downstream, -0.01) << "Waterfall Failure: Gravity/inflow failed to drive downward stream dynamics.";
}
