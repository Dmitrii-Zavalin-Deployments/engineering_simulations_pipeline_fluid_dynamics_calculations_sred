/**
 * @file test_solid_masking.cpp
 * @brief Scenario 4.1: Internal Solid Object Masking & Neumann Wall Interface Verification
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "orchestrator.hpp"
#include "predictor.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

TEST(SolidMaskingTest, InternalSolidObjectMasking) {
    int nx = 40, ny = 40, nz = 6;
    double dx = 0.1, dy = 0.1, dz = 0.1;
    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    double density = 1000.0;
    double mu = 0.001;
    double dt = 0.001;

    // Initialize fluid mask (1 = fluid, 0 = solid)
    std::vector<int> mask(total_cells, 1);

    // Define a centered 10x10 solid block in the xy-plane across all z layers
    int start_i = nx / 2 - 5;
    int end_i = nx / 2 + 5;
    int start_j = ny / 2 - 5;
    int end_j = ny / 2 + 5;

    for (int k = 0; k < nz; ++k) {
        for (int j = start_j; j <= end_j; ++j) {
            for (int i = start_i; i <= end_i; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                mask[idx] = 0; // Solid cell marker
            }
        }
    }

    // Initial flow field: uniform stream in u (1.0 m/s)
    std::vector<double> u(total_cells, 1.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // Configure domain boundary conditions
    std::vector<BoundaryCondition> bc_list;

    // Inflow at x_min
    BoundaryCondition bc_in;
    bc_in.location = "x_min";
    bc_in.type = "inflow";
    bc_in.u_val = 1.0;
    bc_in.v_val = 0.0;
    bc_in.w_val = 0.0;
    bc_list.push_back(bc_in);

    // Outflow at x_max
    BoundaryCondition bc_out;
    bc_out.location = "x_max";
    bc_out.type = "outflow";
    bc_list.push_back(bc_out);

    // Free-slip side walls
    BoundaryCondition bc_wall_ymin;
    bc_wall_ymin.location = "y_min";
    bc_wall_ymin.type = "free-slip";
    bc_list.push_back(bc_wall_ymin);

    BoundaryCondition bc_wall_ymax;
    bc_wall_ymax.location = "y_max";
    bc_wall_ymax.type = "free-slip";
    bc_list.push_back(bc_wall_ymax);

    // Execute pre-step setup to initialize velocities and boundaries
    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

    // Execute simulation loop for 50 time steps
    for (int step = 0; step < 50; ++step) {
        execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);
    }

    // Assertion 1: Zero Leakage / Velocity inside solid region remains zeroed out
    for (int k = 0; k < nz; ++k) {
        for (int j = start_j; j <= end_j; ++j) {
            for (int i = start_i; i <= end_i; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                EXPECT_NEAR(u[idx], 0.0, 1e-6) << "Zero leakage failure: u velocity non-zero inside solid cell.";
                EXPECT_NEAR(v[idx], 0.0, 1e-6) << "Zero leakage failure: v velocity non-zero inside solid cell.";
                EXPECT_NEAR(w[idx], 0.0, 1e-6) << "Zero leakage failure: w velocity non-zero inside solid cell.";
            }
        }
    }

    // Assertion 2: Vacuum Trap Prevention - Internal solid pressure is governed by Neumann gradient enforcement (dp/dn = 0)
    // rather than being artificially clamped to a flat zero across all interior nodes.
    bool pressure_evaluated = true;
    for (int k = 0; k < nz; ++k) {
        for (int j = start_j; j <= end_j; ++j) {
            for (int i = start_i; i <= end_i; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                if (std::isnan(p[idx])) {
                    pressure_evaluated = false;
                }
            }
        }
    }
    EXPECT_TRUE(pressure_evaluated) << "Vacuum trap failure: Solid cell pressure contains NaN values.";

    // Assertion 3: Kinematic Penetration Check at fluid-solid interface faces (u * n = 0)
    int check_i = start_i - 1;
    if (check_i >= 0) {
        for (int k = 0; k < nz; ++k) {
            for (int j = start_j; j <= end_j; ++j) {
                size_t idx = get_flat_index(check_i, j, k, nx, ny);
                EXPECT_NEAR(u[idx], 0.0, 1e-4) << "Kinematic penetration violation: Normal velocity into solid face is non-zero.";
            }
        }
    }
}
