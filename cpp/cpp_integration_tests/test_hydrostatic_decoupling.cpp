/**
 * @file test_hydrostatic_decoupling.cpp
 * @brief Literate Integration Test Suite split into Static Pool Equilibrium 
 *        and Waterfall Inflow Dynamics.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "orchestrator.hpp"
#include "predictor.hpp"
#include "pressure_poisson_solver.hpp"
#include "corrector.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"

using json = nlohmann::json;
using namespace navier_stokes_solver;

class HydrostaticDecouplingDiagnosticTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::ifstream config_stream("cpp/cpp_integration_tests/data/config.json");
        if (config_stream.is_open()) config_stream >> config_json_;

        std::ifstream input_stream("cpp/cpp_integration_tests/data/navier_stokes_input.json");
        if (input_stream.is_open()) input_stream >> input_json_;
    }

    json config_json_;
    json input_json_;
};

// -----------------------------------------------------------------------------
// Test 1: Static Pool Equilibrium (Hydrostatic Balance & Spurious Current Suppression)
// -----------------------------------------------------------------------------
TEST_F(HydrostaticDecouplingDiagnosticTest, StaticPoolEquilibrium) {
    int nx = 9;
    int ny = 9;
    int nz = 9;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    double density = 1000.0; // Fluid density (kg/m^3)
    double mu = 0.001;       // Dynamic viscosity (Pa*s)
    double dt = 0.001;       // Controlled time step size (s)

    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = 500;
    config.poisson_tolerance = 1e-12;

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // Set up a closed fluid mask: 1 for active fluid cells, -1 for wall boundaries.
    std::vector<int> mask(total_cells, 1);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
                    mask[idx] = -1; // Boundary wall
                }
            }
        }
    }

    // Initialize Hydrostatic Pressure Profile (p = -rho * g_y * (y_top - y))
    double gravity_y = -9.81;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                double y_coord = static_cast<double>(j) * dy;
                double y_top = static_cast<double>(ny - 1) * dy;
                p[idx] = -density * gravity_y * (y_top - y_coord);
            }
        }
    }

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

    // Execute Pre-Step Boundary Conditions
    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

    // Execute Predictor Step (Trial Velocities)
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

    double max_v_final = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (mask[idx] == 1) {
            max_v_final = std::max(max_v_final, std::abs(v[idx]));
        }
    }

    // Assertion: Spurious vertical velocity must remain strictly below threshold
    EXPECT_LT(max_v_final, 1e-10) << "Static Pool Failure: Spurious currents generated in equilibrium.";
}

// -----------------------------------------------------------------------------
// Test 2: Waterfall Inflow Dynamics (Gravity-Driven Pouring Flow)
// -----------------------------------------------------------------------------
TEST_F(HydrostaticDecouplingDiagnosticTest, WaterfallInflowDynamics) {
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

    // Initialize an inflow stream at the top center of the domain
    int top_j = ny - 1;
    for (int k = 3; k <= 5; ++k) {
        for (int i = 3; i <= 5; ++i) {
            size_t idx = get_flat_index(i, top_j, k, nx, ny);
            v[idx] = -0.5; // Downward injection velocity
        }
    }

    // Execute Pre-Step Boundary Conditions
    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

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
