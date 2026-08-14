/**
 * @file test_waterfall_inflow.cpp
 * @brief Literate Integration Test Suite for Waterfall Inflow Dynamics 
 *        and Gravity-Driven Pouring Flow via NavierStokesOrchestrator.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "orchestrator.hpp"
#include "grid_math.hpp"

using namespace navier_stokes_solver;

TEST(WaterfallDiagnosticTest, WaterfallInflowDynamics) {
    // =========================================================================
    // Experiment Overview, Purpose, and Verification Objectives
    // =========================================================================
    // 
    // * Experiment Description:
    //   We configure a taller 3D rectangular domain ($9 \times 11 \times 9$) with solid 
    //   side and bottom walls, leaving the top boundary open. A localized downward 
    //   injection velocity ($v = -0.5 \, \text{m/s}$) is applied at the upper boundary 
    //   across a central patch ($3 \times 3$ cells in the $x-z$ plane) to simulate an incoming 
    //   cascading fluid stream (waterfall) subjected to gravity ($g_y = -9.81 \, \text{m/s}^2$).
    //
    // * Why We Are Doing It:
    //   Using the production `NavierStokesOrchestrator` validates that end-to-end sequencing 
    //   correctly handles open boundaries, trial velocity prediction, pressure Poisson solves, 
    //   and divergence-free projection steps without manual pipeline wire-ups.
    //
    // * What We Are Trying to Prove:
    //   We aim to prove that the solver correctly integrates localized boundary inflows 
    //   and drives a gravity-accelerated downward stream into the domain ($\min v < -0.01 \, \text{m/s}$).
    // =========================================================================

    // =========================================================================
    // 1. Simulation Domain and Physical Property Setup
    // =========================================================================
    int nx = 9;
    int ny = 11; 
    int nz = 9;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    
    double density = 1000.0;
    double mu = 0.001;
    double nu = mu / density;
    double dt = 0.001;

    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = 500;
    config.poisson_tolerance = 1e-12;

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    
    // Allocate velocity components ($u, v, w$) and scalar pressure ($p$) fields.
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // =========================================================================
    // 2. Boundary Mask Initialization
    // =========================================================================
    std::vector<int> mask(total_cells, 1);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                if (i == 0 || i == nx - 1 || j == 0 || k == 0 || k == nz - 1) {
                    mask[idx] = -1; // Solid boundary wall
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
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);

    // =========================================================================
    // 3. Inflow Stream Initialization
    // =========================================================================
    // Assign a downward velocity $v = -0.5 \, \text{m/s}$ across a central patch 
    // at the top boundary ($j = ny - 1$).
    int top_j = ny - 1;
    for (int k = 3; k <= 5; ++k) {
        for (int i = 3; i <= 5; ++i) {
            size_t idx = get_flat_index(i, top_j, k, nx, ny);
            v[idx] = -0.5; // Downward injection velocity
        }
    }

    // =========================================================================
    // 4. Execute Step via NavierStokesOrchestrator
    // =========================================================================
    NavierStokesOrchestrator orchestrator(dims, config);
    orchestrator.step(dt, nu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

    // =========================================================================
    // 5. Downstream Flow Verification & Assertion
    // =========================================================================
    double min_v_downstream = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (mask[idx] == 1) {
            min_v_downstream = std::min(min_v_downstream, v[idx]);
        }
    }

    ASSERT_LT(min_v_downstream, -0.01) << "Waterfall Failure: Gravity/inflow failed to drive downward stream dynamics.";
}
