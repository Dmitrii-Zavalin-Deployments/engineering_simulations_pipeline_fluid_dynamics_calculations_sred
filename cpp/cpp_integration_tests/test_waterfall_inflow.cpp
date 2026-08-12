/**
 * @file test_waterfall_inflow.cpp
 * @brief Literate Integration Test Suite for Waterfall Inflow Dynamics 
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
    //   In Eulerian fluid solvers, handling open boundaries and localized mass/momentum 
    //   injection under external body forces is prone to numerical instability or artificial 
    //   damping if boundary conditions and pressure projection steps are improperly sequenced.
    //
    // * What We Are Trying to Prove:
    //   We aim to prove that the solver correctly integrates localized boundary inflows 
    //   with trial velocity prediction, pressure Poisson solve, and divergence-free projection, 
    //   successfully driving and sustaining a gravity-accelerated downward stream into the 
    //   domain ($\min v < -0.01 \, \text{m/s}$).
    // =========================================================================

    // =========================================================================
    // 1. Simulation Domain and Physical Property Setup
    // =========================================================================
    // We define a 3D cartesian grid of dimensions $9 \times 11 \times 9$ (taller vertical 
    // extent to accommodate the falling stream) with uniform mesh spacing 
    // $\Delta x = \Delta y = \Delta z = 1.0 \, \text{m}$.
    int nx = 9;
    int ny = 11; 
    int nz = 9;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    
    // The fluid medium uses standard water properties ($\rho = 1000.0 \, \text{kg/m}^3$, 
    // $\mu = 0.001 \, \text{Pa}\cdot\text{s}$) with a time step $\Delta t = 0.001 \, \text{s}$.
    double density = 1000.0;
    double mu = 0.001;
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
    // We configure the domain enclosure: side and bottom boundaries are marked as 
    // solid walls ($mask = -1$), while the top boundary remains open ($mask = 1$).
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

    // =========================================================================
    // 3. Pre-Step Boundary Condition Enforcement
    // =========================================================================
    // Execute boundary condition setup prior to applying specialized inflow velocities.
    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

    // =========================================================================
    // 4. Inflow Stream Initialization
    // =========================================================================
    // Initialize the waterfall inflow stream *after* the pre-step enforcement to prevent 
    // boundary condition routines from overwriting the injected velocities. We assign a 
    // downward velocity $v = -0.5 \, \text{m/s}$ across a central patch at the top boundary ($j = ny - 1$).
    int top_j = ny - 1;
    for (int k = 3; k <= 5; ++k) {
        for (int i = 3; i <= 5; ++i) {
            size_t idx = get_flat_index(i, top_j, k, nx, ny);
            v[idx] = -0.5; // Downward injection velocity
        }
    }

    // =========================================================================
    // 5. Predictor Step: Trial Velocity Computation
    // =========================================================================
    // Compute trial velocities ($\mathbf{u}^*$) incorporating advection, diffusion, 
    // and gravitational acceleration ($g_y$).
    FluidProperties fluid{mu / density, density};
    compute_trial_velocities(
        dims, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity, mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // =========================================================================
    // 6. Pressure Poisson Equation Formulation & Solve
    // =========================================================================
    // Formulate the divergence source term from trial velocities and solve the 
    // Poisson pressure equation using parallel Red-Black Gauss-Seidel sweeps.
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

    // =========================================================================
    // 7. Corrector Step Projection & Velocity Update
    // =========================================================================
    // Project the velocity field to satisfy continuity using the computed pressure gradient.
    solve_corrector_parallel(
        u, v, w,
        u_star, v_star, w_star,
        p, mask,
        nx, ny, nz, dx, dy, dz,
        dt, density
    );

    // =========================================================================
    // 8. Downstream Flow Verification & Assertion
    // =========================================================================
    // Scan active fluid cells to determine the minimum (most negative) vertical velocity.
    double min_v_downstream = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (mask[idx] == 1) {
            min_v_downstream = std::min(min_v_downstream, v[idx]);
        }
    }

    // Assert that gravity and inflow injection successfully drive a downward stream:
    //     $\min v < -0.01 \, \text{m/s}$
    EXPECT_LT(min_v_downstream, -0.01) << "Waterfall Failure: Gravity/inflow failed to drive downward stream dynamics.";
}
