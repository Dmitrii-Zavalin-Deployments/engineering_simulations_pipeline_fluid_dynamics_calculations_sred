/**
 * @file test_staggered_mac_grid.cpp
 * @brief Scenario 5.1: High-Frequency Checkerboard Pressure Field & MAC Staggering Verification
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "orchestrator.hpp"
#include "corrector.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

TEST(StaggeredMacGridTest, CheckerboardPressureSuppression) {
    int nx = 6, ny = 6, nz = 6;
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;
    double z_min = 0.0, z_max = 1.0;

    double dx = (x_max - x_min) / nx;
    double dy = (y_max - y_min) / ny;
    double dz = (z_max - z_min) / nz;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    double density = 1.0;
    double mu = 0.01;
    double dt = 0.001;

    // Initialize fields
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // Scenario 5.1 Input: Synthetic pressure field with alternating cell-center values
    // P_i,j,k = +100 Pa for even (i+j+k) and -100 Pa for odd (i+j+k)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                if ((i + j + k) % 2 == 0) {
                    p[idx] = 100.0;
                } else {
                    p[idx] = -100.0;
                }
            }
        }
    }

    std::vector<int> mask(total_cells, 1); // All fluid cells for this test
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);
    std::vector<double> gravity = {0.0, 0.0, 0.0};

    std::vector<BoundaryCondition> bc_list;
    BoundaryCondition bc_wall;
    bc_wall.location = "wall";
    bc_wall.type = "no-slip";
    bc_wall.u_val = 0.0;
    bc_wall.v_val = 0.0;
    bc_wall.w_val = 0.0;
    bc_list.push_back(bc_wall);

    SolverConfig config{2000, 1e-8, density};

    // Execute direct corrector projection step to evaluate MAC staggering response to checkerboard gradients
    solve_corrector_parallel(u, v, w, u, v, w, p, mask, nx, ny, nz, dx, dy, dz, dt, density);

    // Assertion 1: Verify that velocities responded immediately to adjacent cell pressure differences (non-zero u, v, w updates)
    bool velocities_responded = false;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (std::abs(u[idx]) > 1e-12 || std::abs(v[idx]) > 1e-12 || std::abs(w[idx]) > 1e-12) {
            velocities_responded = true;
            break;
        }
    }
    ASSERT_TRUE(velocities_responded) << "MAC grid failure: Velocities did not respond to checkerboard pressure gradients.";

    // Assertion 2: Verify pressure field remains bounded and finite
    bool pressure_bounded = true;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (std::isnan(p[idx]) || std::abs(p[idx]) > 10000.0) {
            pressure_bounded = false;
            break;
        }
    }
    ASSERT_TRUE(pressure_bounded) << "Pressure instability detected during checkerboard suppression.";
}
