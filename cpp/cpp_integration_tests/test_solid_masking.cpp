/**
 * @file test_solid_masking.cpp
 * @brief Scenario 4.1: Internal Solid Object Masking & Boundary Enforcement Verification
 * 
 * LITERATE TESTING NARRATIVE & MATHEMATICAL FORMULATION:
 * ---------------------------------------------------------------------------------
 * This integration test verifies that the Navier-Stokes orchestrator correctly 
 * respects internal solid boundaries and domain walls via binary mask arrays,
 * ensuring non-fluid cells are robustly clamped to zero velocity.
 * 
 * Grid spacing increments are defined as:
 *     dx = (x_max - x_min) / nx
 *     dy = (y_max - y_min) / ny
 *     dz = (z_max - z_min) / nz
 * 
 * Mask convention:
 *     mask[idx] = 1 -> Fluid cell (subject to flow equations and projection steps)
 *     mask[idx] = 0 -> Solid cell (clamped to zero velocity)
 *     mask[idx] = -1 -> Wall boundary cell (clamped to zero velocity)
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include "orchestrator.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

TEST(SolidMaskingTest, InternalSolidObjectMasking) {
    int nx = 4, ny = 4, nz = 4;
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;
    double z_min = 0.0, z_max = 1.0;

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

    std::vector<int> mask = {
        // k = 0 (solid boundary layer)
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        // k = 1 (fluid core with internal solid obstacle at index [1,1,1])
        0, 0, 0, 0,
        0, 0, 1, 0,
        0, 1, 1, 0,
        0, 0, 0, 0,

        // k = 2 (fluid core with internal solid obstacle)
        0, 0, 0, 0,
        0, 0, 1, 0,
        0, 1, 1, 0,
        0, 0, 0, 0,

        // k = 3 (solid boundary layer)
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };
    assert(mask.size() == total_cells);

    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (mask[idx] == 1) {
            u[idx] = 1.0;
            v[idx] = 0.0;
            w[idx] = 0.0;
        } else {
            u[idx] = 0.0;
            v[idx] = 0.0;
            w[idx] = 0.0;
        }
    }

    size_t internal_solid_idx = get_flat_index(1, 1, 1, nx, ny);

    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);
    std::vector<double> gravity = {0.0, -9.81, 0.0};

    std::vector<BoundaryCondition> bc_list;
    BoundaryCondition bc_wall;
    bc_wall.location = "wall";
    bc_wall.type = "no-slip";
    bc_wall.u_val = 0.0;
    bc_wall.v_val = 0.0;
    bc_wall.w_val = 0.0;
    bc_list.push_back(bc_wall);

    SolverConfig config{2000, 1e-8, density};
    NavierStokesOrchestrator orchestrator(dims, config);

    for (int step = 0; step < 10; ++step) {
        orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);
    }

    // Assertion 1: Verify internal solid cells are properly clamped to zero velocity
    if (mask[internal_solid_idx] == 0) {
        assert(std::abs(u[internal_solid_idx]) < 1e-6);
        assert(std::abs(v[internal_solid_idx]) < 1e-6);
        assert(std::abs(w[internal_solid_idx]) < 1e-6);

        ASSERT_NEAR(u[internal_solid_idx], 0.0, 1e-6) << "Internal solid cell velocity not zeroed.";
        ASSERT_NEAR(v[internal_solid_idx], 0.0, 1e-6) << "Internal solid cell velocity not zeroed.";
        ASSERT_NEAR(w[internal_solid_idx], 0.0, 1e-6) << "Internal solid cell velocity not zeroed.";
    }

    // Assertion 2: Verify domain boundary / wall cells are properly enforced to 0.0 (no-slip)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1) {
                    size_t idx = get_flat_index(i, j, k, nx, ny);
                    assert(std::abs(u[idx]) < 1e-6);
                    assert(std::abs(v[idx]) < 1e-6);
                    assert(std::abs(w[idx]) < 1e-6);

                    ASSERT_NEAR(u[idx], 0.0, 1e-6);
                    ASSERT_NEAR(v[idx], 0.0, 1e-6);
                    ASSERT_NEAR(w[idx], 0.0, 1e-6);
                }
            }
        }
    }

    // Assertion 3: Pressure stability inside solid region
    bool pressure_valid = true;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (mask[idx] == 0 && std::isnan(p[idx])) {
            pressure_valid = false;
            break;
        }
    }
    assert(pressure_valid);
    ASSERT_TRUE(pressure_valid) << "Vacuum trap failure: Solid cell pressure contains NaN values.";
}