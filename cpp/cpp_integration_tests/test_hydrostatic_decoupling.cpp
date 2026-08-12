/**
 * @file test_hydrostatic_decoupling.cpp
 * @brief Step-by-Step Diagnostic Integration Test for Hydrostatic Decoupling in a Water Column.
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

TEST_F(HydrostaticDecouplingDiagnosticTest, StepByStepWaterColumnIsolation) {
    int nx = 9;
    int ny = 9;
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

    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> rhs(total_cells, 0.0);

    // Set up a simple fluid mask (1 for fluid, 0 for solid, -1 for wall/boundary)
    std::vector<int> mask(total_cells, 1);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
                    mask[idx] = -1; // Boundary
                }
            }
        }
    }

    std::vector<BoundaryCondition> bc_list;
    BoundaryCondition bc;
    bc.location = "wall";
    bc.type = "velocity";
    bc.u_val = 0.0; bc.v_val = 0.0; bc.w_val = 0.0; bc.scalar_p = 0.0;
    bc_list.push_back(bc);

    std::vector<double> gravity = {0.0, -9.81, 0.0};
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, density * (-9.81));
    std::vector<double> fz(total_cells, 0.0);

    std::cout << "\n=== DIAGNOSTIC CHECKPOINT 1: PRE-STEP ===" << std::endl;
    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);
    SUCCEED();

    std::cout << "\n=== DIAGNOSTIC CHECKPOINT 2: PREDICTOR STEP ===" << std::endl;
    FluidProperties fluid{mu / density, density};
    compute_trial_velocities(
        dims, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity, mask,
        u_star.data(), v_star.data(), w_star.data()
    );
    
    double max_v_star = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (mask[idx] == 1) {
            max_v_star = std::max(max_v_star, std::abs(v_star[idx]));
        }
    }
    std::cout << "Max trial velocity v* after predictor: " << max_v_star << std::endl;

    std::cout << "\n=== DIAGNOSTIC CHECKPOINT 3: PRESSURE POISSON RHS & SOLVE ===" << std::endl;
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
        config.max_poisson_iterations, config.poisson_tolerance
    );

    double max_p = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        max_p = std::max(max_p, std::abs(p[idx]));
    }
    std::cout << "Max pressure computed by Poisson solver: " << max_p << std::endl;

    std::cout << "\n=== DIAGNOSTIC CHECKPOINT 4: CORRECTOR STEP ===" << std::endl;
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
    std::cout << "Max velocity v after corrector projection: " << max_v_final << std::endl;

    EXPECT_LT(max_v_final, 1e-10) << "Failure isolated: Spurious currents generated during projection step.";
}
