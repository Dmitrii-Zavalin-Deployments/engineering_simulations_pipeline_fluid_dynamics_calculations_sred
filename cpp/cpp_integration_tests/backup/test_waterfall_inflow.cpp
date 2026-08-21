/**
 * @file test_waterfall_inflow.cpp
 * @brief Literate Integration Test Suite for Waterfall Inflow Dynamics 
 *        and Gravity-Driven Pouring Flow via NavierStokesOrchestrator.
 * 
 * LITERATE TESTING NARRATIVE & MATHEMATICAL FORMULATION:
 * ---------------------------------------------------------------------------------
 * Experiment Description:
 * We configure a taller 3D rectangular domain (9 x 11 x 9) with solid 
 * side and bottom walls, leaving the top boundary open. A localized downward 
 * injection velocity (v = -0.5 m/s) is applied at the upper boundary 
 * across a central patch (3 x 3 cells in the x-z plane) to simulate an incoming 
 * cascading fluid stream (waterfall) subjected to gravity (g_y = -9.81 m/s^2).
 * 
 * Why We Are Doing It:
 * Using the production NavierStokesOrchestrator validates that end-to-end sequencing 
 * correctly handles open boundaries, trial velocity prediction, pressure Poisson solves, 
 * and divergence-free projection steps without manual pipeline wire-ups.
 * 
 * What We Are Trying to Prove:
 * We aim to prove that the solver correctly integrates localized boundary inflows 
 * and drives a gravity-accelerated downward stream into the domain (min v < -0.01 m/s).
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include "orchestrator.hpp"
#include "grid_math.hpp"

using namespace navier_stokes_solver;

TEST(WaterfallDiagnosticTest, WaterfallInflowDynamics) {
    // We define a taller 3D cartesian grid of dimensions 9 x 11 x 9 with uniform mesh spacing dx = dy = dz = 1.0 m.
    int nx = 9;
    int ny = 11; 
    int nz = 9;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    assert(nx > 0 && ny > 0 && nz > 0);
    assert(dx > 0.0 && dy > 0.0 && dz > 0.0);
    
    // We define the fluid properties (density = 1000.0 kg/m^3) and simulation time step.
    double density = 1000.0;
    double mu = 0.001;
    double nu = mu / density;
    double dt = 0.001;

    assert(density > 0.0);
    assert(dt > 0.0);

    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = 500;
    config.poisson_tolerance = 1e-12;

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    
    // We allocate velocity components (u, v, w) and scalar pressure (p) fields initialized to zero.
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // We initialize the boundary mask:
    //  1 indicates an active fluid interior cell.
    // -1 indicates a solid boundary wall.
    std::vector<int> mask(total_cells, 1);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                // We leave the top boundary open (j != ny - 1), but set solid side and bottom walls.
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
    assert(!bc_list.empty());

    // We apply a downward gravitational acceleration of -9.81 m/s^2.
    std::vector<double> gravity = {0.0, -9.81, 0.0};
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);

    // We initialize an incoming cascading stream (waterfall):
    // Assign a downward velocity v = -0.5 m/s across a central patch 
    // at the top boundary (j = ny - 1).
    int top_j = ny - 1;
    for (int k = 3; k <= 5; ++k) {
        for (int i = 3; i <= 5; ++i) {
            size_t idx = get_flat_index(i, top_j, k, nx, ny);
            v[idx] = -0.5; // Downward injection velocity
        }
    }

    // We instantiate the orchestrator and execute one full simulation step.
    NavierStokesOrchestrator orchestrator(dims, config);
    orchestrator.step(dt, nu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

    // We evaluate downstream flow acceleration.
    // Measure the minimum vertical velocity across active fluid cells to ensure the waterfall pushes downward.
    double min_v_downstream = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (mask[idx] == 1) {
            min_v_downstream = std::min(min_v_downstream, v[idx]);
        }
    }

    // We assert that the solver correctly integrates localized boundary inflows 
    // and drives a gravity-accelerated downward stream into the domain (min v < -0.01 m/s).
    assert(min_v_downstream < -0.01);
    ASSERT_LT(min_v_downstream, -0.01) << "Waterfall Failure: Gravity/inflow failed to drive downward stream dynamics.";
}
