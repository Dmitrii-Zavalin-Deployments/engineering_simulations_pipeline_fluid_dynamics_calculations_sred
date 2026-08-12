/**
 * @file test_solid_masking.cpp
 * @brief Scenario 4.1: Internal Solid Object Masking & Neumann Wall Interface Verification
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "orchestrator.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

TEST(SolidMaskingTest, InternalSolidObjectMasking) {
    // Grid parameters matching cpp_integration_tests/data/navier_stokes_input.json
    int nx = 4, ny = 4, nz = 4;
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;
    double z_min = 0.0, z_max = 1.0;

    double dx = (x_max - x_min) / nx;
    double dy = (y_max - y_min) / ny;
    double dz = (z_max - z_min) / nz;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // Physical and numerical parameters from dataset
    double density = 1.0;
    double mu = 0.01;
    double dt = 0.001;

    // Fluid mask (64 elements: 1 = fluid, 0 = solid) matching input JSON structure
    std::vector<int> mask = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 1, 1, 0,
        0, 1, 1, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 1, 1, 0,
        0, 1, 1, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    // Initial flow field state
    std::vector<double> u(total_cells, 1.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // External force and gravity vectors
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);
    std::vector<double> gravity = {0.0, -9.81, 0.0};

    // Domain boundary conditions
    std::vector<BoundaryCondition> bc_list;

    BoundaryCondition bc_wall;
    bc_wall.location = "wall";
    bc_wall.type = "no-slip";
    bc_wall.u_val = 0.0;
    bc_wall.v_val = 0.0;
    bc_wall.w_val = 0.0;
    bc_list.push_back(bc_wall);

    // Instantiate orchestrator using configuration from config.json
    SolverConfig config{2000, 1e-8, density};
    NavierStokesOrchestrator orchestrator(dims, config);

    // Execute 10 solver steps (total time 0.01s / dt 0.001s)
    for (int step = 0; step < 10; ++step) {
        orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);
    }

    // Assertion 1: Zero Leakage - Velocity inside all solid cells must be 0.0
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                if (mask[idx] == 0) {
                    EXPECT_NEAR(u[idx], 0.0, 1e-6) << "Zero leakage failure (u) at (" << i << ", " << j << ", " << k << ")";
                    EXPECT_NEAR(v[idx], 0.0, 1e-6) << "Zero leakage failure (v) at (" << i << ", " << j << ", " << k << ")";
                    EXPECT_NEAR(w[idx], 0.0, 1e-6) << "Zero leakage failure (w) at (" << i << ", " << j << ", " << k << ")";
                }
            }
        }
    }

    // Assertion 2: Vacuum Trap Prevention - Pressure in solid region must not evaluate to NaN
    bool pressure_valid = true;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (mask[idx] == 0 && std::isnan(p[idx])) {
            pressure_valid = false;
            break;
        }
    }
    EXPECT_TRUE(pressure_valid) << "Vacuum trap failure: Solid cell pressure contains NaN values.";
}
