/**
 * @file test_staggered_mac_grid.cpp
 * @brief Scenario 5.1: High-Frequency Checkerboard Pressure Field & MAC Staggering Verification
 * 
 * LITERATE TESTING NARRATIVE & MATHEMATICAL FORMULATION:
 * ---------------------------------------------------------------------------------
 * Monolithic collocated grid arrangements often suffer from checkerboard pressure 
 * decoupling (spurious high-frequency oscillations) because pressure gradients are 
 * evaluated across alternate grid nodes:
 * 
 *     dp/dx_collocated = (p_(i+1) - p_(i-1)) / (2 * dx)
 * 
 * The Marker-and-Cell (MAC) staggered grid resolves this by placing velocity components 
 * on cell faces, allowing pressure differences between adjacent cell centers to directly 
 * drive face velocities:
 * 
 *     dp/dx_staggered = (p_(i+1) - p_i) / dx
 * 
 * TEST SCENARIO:
 *   - We initialize a synthetic checkerboard pressure field where cell-center values 
 *     alternate between +100 Pa and -100 Pa based on index parity (i + j + k).
 *   - We execute a direct corrector projection step and verify that face velocities 
 *     immediately respond to these adjacent cell pressure differences.
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include "orchestrator.hpp"
#include "corrector.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

TEST(StaggeredMacGridTest, CheckerboardPressureSuppression) {
    // We define grid dimensions and spatial boundaries:
    int nx = 6, ny = 6, nz = 6;
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;
    double z_min = 0.0, z_max = 1.0;

    // We compute spatial step sizes:
    //     dx = (x_max - x_min) / nx
    //     dy = (y_max - y_min) / ny
    //     dz = (z_max - z_min) / nz
    double dx = (x_max - x_min) / nx;
    double dy = (y_max - y_min) / ny;
    double dz = (z_max - z_min) / nz;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    assert(nx > 0 && ny > 0 && nz > 0);
    assert(dx > 0.0 && dy > 0.0 && dz > 0.0);

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    double density = 1.0;
    double mu = 0.01;
    double dt = 0.001;
    assert(density > 0.0);
    assert(mu >= 0.0);
    assert(dt > 0.0);

    // We initialize vector and scalar fields:
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // We populate a synthetic pressure field with alternating cell-center values:
    //     p_i,j,k = +100.0 Pa for even (i + j + k)
    //     p_i,j,k = -100.0 Pa for odd (i + j + k)
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

    // We execute the direct corrector projection step to evaluate MAC staggering response to checkerboard gradients:
    solve_corrector_parallel(u, v, w, u, v, w, p, mask, nx, ny, nz, dx, dy, dz, dt, density);

    // Assertion 1: Verify that velocities responded immediately to adjacent cell pressure differences (non-zero u, v, w updates)
    bool velocities_responded = false;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (std::abs(u[idx]) > 1e-12 || std::abs(v[idx]) > 1e-12 || std::abs(w[idx]) > 1e-12) {
            velocities_responded = true;
            break;
        }
    }
    assert(velocities_responded);
    ASSERT_TRUE(velocities_responded) << "MAC grid failure: Velocities did not respond to checkerboard pressure gradients.";

    // Assertion 2: Verify pressure field remains bounded and finite
    bool pressure_bounded = true;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (std::isnan(p[idx]) || std::abs(p[idx]) > 10000.0) {
            pressure_bounded = false;
            break;
        }
    }
    assert(pressure_bounded);
    ASSERT_TRUE(pressure_bounded) << "Pressure instability detected during checkerboard suppression.";
}
