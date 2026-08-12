/**
 * @file test_static_pool_equilibrium.cpp
 * @brief Literate Integration Test Suite for Static Pool Equilibrium 
 *        and Spurious Current Suppression.
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

TEST(StaticPoolDiagnosticTest, StaticPoolEquilibrium) {
    // =========================================================================
    // Experiment Overview, Purpose, and Verification Objectives
    // =========================================================================
    // 
    // * Experiment Description:
    //   We configure a closed, rigid 3D rectangular fluid container filled with an 
    //   incompressible fluid under a constant downward gravitational field ($g_y = -9.81 \, \text{m/s}^2$). 
    //   The pressure field is initialized explicitly to satisfy the hydrostatic balance equation:
    //       $p(y) = -\rho g_y (y_{\text{top}} - y)$
    //
    // * Why We Are Doing It:
    //   In fractional-step Navier-Stokes solvers, numerical inconsistencies between the pressure 
    //   gradient discretization and gravity body force terms frequently induce artificial 
    //   acceleration. This flaw causes parasitic currents (spurious velocities) to spontaneously 
    //   emerge in a fluid that should otherwise remain completely static.
    //
    // * What We Are Trying to Prove:
    //   We aim to verify that the complete projection pipeline—comprising trial velocity prediction, 
    //   Poisson pressure solve, and divergence-free velocity correction—maintains exact hydrostatic 
    //   equilibrium. Specifically, we prove that residual fluid velocities remain strictly bounded 
    //   below the spatial discretization truncation error threshold ($\max |v| < 6 \times 10^{-3} \, \text{m/s}$).
    // =========================================================================

    // =========================================================================
    // 1. Simulation Domain and Physical Property Setup
    // =========================================================================
    // We define a 3D cartesian grid of dimensions $9 \times 9 \times 9$ with 
    // uniform mesh spacing $\Delta x = \Delta y = \Delta z = 1.0 \, \text{m}$.
    int nx = 9;
    int ny = 9;
    int nz = 9;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    
    // The fluid medium is water, characterized by a density $\rho = 1000.0 \, \text{kg/m}^3$ 
    // and dynamic viscosity $\mu = 0.001 \, \text{Pa}\cdot\text{s}$. The temporal discretization 
    // uses a controlled step size $\Delta t = 0.001 \, \text{s}$.
    double density = 1000.0; 
    double mu = 0.001;       
    double dt = 0.001;       

    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = 500;
    config.poisson_tolerance = 1e-12;

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    
    // Allocate velocity components ($u, v, w$) and scalar pressure ($p$) fields initialized to zero.
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // =========================================================================
    // 2. Boundary Mask Initialization
    // =========================================================================
    // We configure a closed rigid container. The mask array marks active fluid interior 
    // cells with $1$ and boundary wall cells with $-1$.
    std::vector<int> mask(total_cells, 1);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
                    mask[idx] = -1; // Rigid boundary wall
                }
            }
        }
    }

    // =========================================================================
    // 3. Hydrostatic Pressure Profile Initialization
    // =========================================================================
    // To maintain a static equilibrium under gravity ($g_y = -9.81 \, \text{m/s}^2$), 
    // the pressure field must balance the body force exactly according to the hydrostatic law:
    //     $p(y) = -\rho g_y (y_{\text{top}} - y)$
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

    // =========================================================================
    // 4. Pre-Step Boundary Condition Enforcement
    // =========================================================================
    // Execute boundary condition setup prior to evaluating momentum equations.
    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

    // =========================================================================
    // 5. Predictor Step: Trial Velocity Computation
    // =========================================================================
    // Compute intermediate trial velocities ($\mathbf{u}^*$) accounting for advection, 
    // diffusion, and external gravity body forces without pressure gradient coupling.
    FluidProperties fluid{mu / density, density};
    compute_trial_velocities(
        dims, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity, mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // =========================================================================
    // 6. Pressure Poisson Equation Formulation & Red-Black Gauss-Seidel Solve
    // =========================================================================
    // The divergence of the trial velocity field acts as the source term for the 
    // pressure correction Poisson equation:
    //     $\nabla^2 p^{n+1} = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$
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

    // Solve the Poisson equation iteratively using parallelized Red-Black Gauss-Seidel sweeps.
    solve_poisson_red_black_parallel(
        p, rhs, mask, bc_list,
        nx, ny, nz, dx, dy, dz,
        config.max_poisson_iterations, config.poisson_tolerance,
        config.density, gravity
    );

    // =========================================================================
    // 7. Corrector Step Projection & Final Velocity Update
    // =========================================================================
    // Project the velocity field divergence-free by subtracting the pressure gradient:
    //     $\mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla p^{n+1}$
    solve_corrector_parallel(
        u, v, w,
        u_star, v_star, w_star,
        p, mask,
        nx, ny, nz, dx, dy, dz,
        dt, density
    );

    // =========================================================================
    // 8. Equilibrium Verification & Assertion
    // =========================================================================
    // Measure the maximum residual vertical velocity magnitude across active fluid cells.
    double max_v_final = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (mask[idx] == 1) {
            max_v_final = std::max(max_v_final, std::abs(v[idx]));
        }
    }

    // Assert that spurious velocity currents remain strictly below the discretization error threshold:
    //     $\max |v| < 6 \times 10^{-3} \, \text{m/s}$
    EXPECT_LT(max_v_final, 6e-3) << "Static Pool Failure: Spurious currents generated in equilibrium.";
}
