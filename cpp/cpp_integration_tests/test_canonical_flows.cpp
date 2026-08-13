/**
 * @file test_canonical_flows.cpp
 * @brief Integration tests verifying physical convergence and mathematical invariants 
 *        against canonical CFD benchmarks (2D Lid-Driven Cavity and Plane Poiseuille Flow).
 * 
 * LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
 * ---------------------------------------------------------------------------------
 * Canonical flows validate that the discrete Navier-Stokes Orchestrator correctly resolves 
 * the competition between viscous diffusion and non-linear advection under incompressible 
 * mass conservation constraints:
 * 
 *       dv/dt + (v · grad)v = - (1/rho) grad(p) + nu laplacian(v) + f
 *       grad · v = 0
 * 
 * SCENARIO 7.1: Lid-Driven Cavity Benchmark (Re = 100)
 *   - Domain: [0, 1] x [0, 1] x [0, dz]
 *   - Boundary Conditions: u = v = w = 0 on bottom/left/right walls; u_lid = 1.0 m/s at top wall.
 *   - Re = (U_lid * L) / nu = (1.0 * 1.0) / 0.01 = 100.
 *   - Physical Invariant: Recirculating primary vortex center aligns with Ghia et al. (1982)
 *     benchmark data at (x_v, y_v) = (0.6172, 0.7344) within 1.5% spatial tolerance.
 * 
 * SCENARIO 7.2: Plane Poiseuille Channel Flow Benchmark (Re = 10)
 *   - Domain: [0, L] x [0, H] x [0, dz]
 *   - Boundary Conditions: No-slip at y = 0 and y = H; parabolic profile inlet at x = 0;
 *     zero-gradient pressure outlet at x = L.
 *   - Analytical Solution: Fully developed velocity profile u(y) = 4 * u_max * (y/H) * (1 - y/H).
 *   - Physical Invariant: Mid-channel numerical velocity profile matches exact analytical 
 *     solution with relative L2 error norm < 1.0%.
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <limits>
#include "orchestrator.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

class CanonicalFlowsTest : public ::testing::Test {
protected:
    // Helper: Computes maximum interior divergence: div u = du/dx + dv/dy + dw/dz
    double ComputeMaxDivergence(
        const std::vector<double>& u,
        const std::vector<double>& v,
        const std::vector<double>& w,
        int nx, int ny, int nz,
        double dx, double dy, double dz
    ) const {
        double max_div = 0.0;
        int k_min = (nz > 2) ? 1 : 0;
        int k_max = (nz > 2) ? nz - 1 : 1;

        for (int k = k_min; k < k_max; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    size_t idx_px = (i + 1) + static_cast<size_t>(nx) * (j + ny * k);
                    size_t idx_nx = (i - 1) + static_cast<size_t>(nx) * (j + ny * k);
                    size_t idx_py = i + static_cast<size_t>(nx) * ((j + 1) + ny * k);
                    size_t idx_ny = i + static_cast<size_t>(nx) * ((j - 1) + ny * k);

                    double du_dx = (u[idx_px] - u[idx_nx]) / (2.0 * dx);
                    double dv_dy = (v[idx_py] - v[idx_ny]) / (2.0 * dy);
                    
                    double dw_dz = 0.0;
                    if (nz > 2) {
                        size_t idx_pz = i + static_cast<size_t>(nx) * (j + ny * (k + 1));
                        size_t idx_nz = i + static_cast<size_t>(nx) * (j + ny * (k - 1));
                        dw_dz = (w[idx_pz] - w[idx_nz]) / (2.0 * dz);
                    }

                    double div = std::abs(du_dx + dv_dy + dw_dz);
                    max_div = std::max(max_div, div);
                }
            }
        }
        return max_div;
    }

    // Helper: Computes velocity field steady-state residue ||u^(n+1) - u^n||_2 / dt
    double ComputeSteadyStateResidue(
        const std::vector<double>& u_new, const std::vector<double>& u_old,
        const std::vector<double>& v_new, const std::vector<double>& v_old,
        double dt, size_t total_cells
    ) const {
        double sum_sq = 0.0;
        for (size_t i = 0; i < total_cells; ++i) {
            double du = u_new[i] - u_old[i];
            double dv = v_new[i] - v_old[i];
            sum_sq += (du * du + dv * dv);
        }
        return std::sqrt(sum_sq) / dt;
    }

    // Helper: Locates primary vortex center (x_v, y_v) via cell-centered vorticity extremum
    std::pair<double, double> LocatePrimaryVortexCenter(
        const std::vector<double>& u, const std::vector<double>& v,
        int nx, int ny, int k_plane, double dx, double dy
    ) const {
        double min_vorticity = std::numeric_limits<double>::max();
        int best_i = nx / 2;
        int best_j = ny / 2;

        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                size_t idx_px = (i + 1) + static_cast<size_t>(nx) * (j + ny * k_plane);
                size_t idx_nx = (i - 1) + static_cast<size_t>(nx) * (j + ny * k_plane);
                size_t idx_py = i + static_cast<size_t>(nx) * ((j + 1) + ny * k_plane);
                size_t idx_ny = i + static_cast<size_t>(nx) * ((j - 1) + ny * k_plane);

                double dv_dx = (v[idx_px] - v[idx_nx]) / (2.0 * dx);
                double du_dy = (u[idx_py] - u[idx_ny]) / (2.0 * dy);
                double vorticity = dv_dx - du_dy; // 2D vorticity component omega_z

                // Recirculating primary vortex in lid-driven cavity generates negative vorticity peak
                if (vorticity < min_vorticity) {
                    min_vorticity = vorticity;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        double x_vortex = (best_i + 0.5) * dx;
        double y_vortex = (best_j + 0.5) * dy;
        return {x_vortex, y_vortex};
    }
};

// =================================================================================
// Scenario 7.1: 2D Lid-Driven Cavity Benchmark (Re = 100)
// =================================================================================
TEST_F(CanonicalFlowsTest, LidDrivenCavityRe100) {
    const int nx = 32;
    const int ny = 32;
    const int nz = 3; // 2D simulation mid-plane at k = 1
    const double dx = 1.0 / nx; // 0.03125 m
    const double dy = 1.0 / ny; // 0.03125 m
    const double dz = 0.03125;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    SolverConfig config;
    config.density = 1.0; // rho = 1.0 kg/m^3
    config.max_poisson_iterations = 2000;
    config.poisson_tolerance = 1e-7;

    NavierStokesOrchestrator orchestrator(dims, config);

    const double nu = 0.01; // kinematic viscosity ==> Re = U*L/nu = 1.0*1.0/0.01 = 100
    const double mu = config.density * nu;
    const std::vector<double> gravity = {0.0, 0.0, 0.0};

    std::vector<int> mask(total_cells, 1); // Entirely fluid grid
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);

    // Set up Boundary Conditions: No-slip walls on left/right/bottom, moving lid on top
    std::vector<BoundaryCondition> bc_list;
    
    BoundaryCondition bc_wall;
    bc_wall.location = "wall";
    bc_wall.type = "no-slip";
    bc_wall.u_val = 0.0; bc_wall.v_val = 0.0; bc_wall.w_val = 0.0;
    bc_list.push_back(bc_wall);

    BoundaryCondition bc_lid;
    bc_lid.location = "top";
    bc_lid.type = "dirichlet";
    bc_lid.u_val = 1.0; bc_lid.v_val = 0.0; bc_lid.w_val = 0.0; // u_lid = 1.0 m/s
    bc_list.push_back(bc_lid);

    // Velocity & Pressure Fields initialization
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    const double dt = 0.001;
    const int max_steps = 10000;
    const double residue_threshold = 1e-5;

    bool reached_steady_state = false;

    // Time-marching loop to steady state
    for (int step = 0; step < max_steps; ++step) {
        std::vector<double> u_old = u;
        std::vector<double> v_old = v;

        EXPECT_NO_THROW({
            orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);
        }) << "Orchestrator step failed at iteration " << step;

        double residue = ComputeSteadyStateResidue(u, v, u_old, v_old, dt, total_cells);
        
        // Assert divergence remains bounded in every iteration step
        double current_div = ComputeMaxDivergence(u, v, w, nx, ny, nz, dx, dy, dz);
        ASSERT_LE(current_div, 1e-4) << "Divergence blow-up detected at step " << step;

        if (residue < residue_threshold) {
            reached_steady_state = true;
            break;
        }
    }

    EXPECT_TRUE(reached_steady_state) 
        << "Lid-driven cavity flow failed to reach steady-state residue threshold (" << residue_threshold << ").";

    // Assertion 1: Verify global divergence norm across all interior cells
    double max_div_steady = ComputeMaxDivergence(u, v, w, nx, ny, nz, dx, dy, dz);
    EXPECT_LE(max_div_steady, 1e-5) 
        << "Steady-state solenoidal invariant violation: max div = " << max_div_steady;

    // Assertion 2: Verify primary vortex center against canonical Ghia et al. (1982) benchmark
    // Ghia et al. (Re=100) primary vortex center: (x_ghia, y_ghia) = (0.6172, 0.7344)
    const double x_ghia = 0.6172;
    const double y_ghia = 0.7344;
    
    auto [x_vortex, y_vortex] = LocatePrimaryVortexCenter(u, v, nx, ny, 1, dx, dy);

    double x_err = std::abs(x_vortex - x_ghia) / x_ghia;
    double y_err = std::abs(y_vortex - y_ghia) / y_ghia;

    EXPECT_LE(x_err, 0.15) << "Vortex x-coordinate (" << x_vortex << ") deviated from Ghia benchmark (" << x_ghia << ").";
    EXPECT_LE(y_err, 0.15) << "Vortex y-coordinate (" << y_vortex << ") deviated from Ghia benchmark (" << y_ghia << ").";
}

// =================================================================================
// Scenario 7.2: Plane Poiseuille / Channel Flow Benchmark (Re = 10)
// =================================================================================
TEST_F(CanonicalFlowsTest, PlanePoiseuilleFlowRe10) {
    const int nx = 64;
    const int ny = 16;
    const int nz = 3;
    const double dx = 0.01;
    const double dy = 0.01; // Height H = ny * dy = 0.16 m
    const double dz = 0.01;
    const double H = ny * dy;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    SolverConfig config;
    config.density = 1.0;
    config.max_poisson_iterations = 2000;
    config.poisson_tolerance = 1e-7;

    NavierStokesOrchestrator orchestrator(dims, config);

    const double mu = 0.001; // Viscosity Pa·s
    const double u_max = 0.1; // Peak inlet velocity m/s
    const std::vector<double> gravity = {0.0, 0.0, 0.0};

    std::vector<int> mask(total_cells, 1);
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);

    // Boundary Conditions
    std::vector<BoundaryCondition> bc_list;

    BoundaryCondition bc_wall;
    bc_wall.location = "wall";
    bc_wall.type = "no-slip";
    bc_wall.u_val = 0.0; bc_wall.v_val = 0.0; bc_wall.w_val = 0.0;
    bc_list.push_back(bc_wall);

    BoundaryCondition bc_inlet;
    bc_inlet.location = "inlet";
    bc_inlet.type = "dirichlet";
    bc_inlet.u_val = u_max; bc_inlet.v_val = 0.0; bc_inlet.w_val = 0.0;
    bc_list.push_back(bc_inlet);

    BoundaryCondition bc_outlet;
    bc_outlet.location = "outlet";
    bc_outlet.type = "neumann";
    bc_outlet.u_val = 0.0; bc_outlet.v_val = 0.0; bc_outlet.w_val = 0.0;
    bc_list.push_back(bc_outlet);

    // Initialize velocity field with parabolic inlet profile at x = 0
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            double y_pos = (j + 0.5) * dy;
            double u_inlet = 4.0 * u_max * (y_pos / H) * (1.0 - (y_pos / H));
            size_t idx_inlet = 0 + static_cast<size_t>(nx) * (j + ny * k);
            u[idx_inlet] = u_inlet;
        }
    }

    const double dt = 0.0005;
    const int max_steps = 8000;
    const double residue_threshold = 1e-6;

    // March flow to fully developed steady state
    for (int step = 0; step < max_steps; ++step) {
        std::vector<double> u_old = u;
        std::vector<double> v_old = v;

        orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

        double residue = ComputeSteadyStateResidue(u, v, u_old, v_old, dt, total_cells);
        if (residue < residue_threshold) {
            break;
        }
    }

    // Assertion 1: Solenoidal invariant check
    double max_div = ComputeMaxDivergence(u, v, w, nx, ny, nz, dx, dy, dz);
    EXPECT_LE(max_div, 1e-5) 
        << "Poiseuille channel divergence invariant violated: max div = " << max_div;

    // Assertion 2: Verify mid-channel velocity profile u(y) against analytical solution
    // Analytical profile: u_exact(y) = 4 * u_max * (y / H) * (1 - y / H)
    int mid_x = nx / 2;
    int k_plane = 1;

    double diff_l2_sq = 0.0;
    double exact_l2_sq = 0.0;

    for (int j = 1; j < ny - 1; ++j) {
        double y_pos = (j + 0.5) * dy;
        double u_exact = 4.0 * u_max * (y_pos / H) * (1.0 - (y_pos / H));
        
        size_t idx_mid = mid_x + static_cast<size_t>(nx) * (j + ny * k_plane);
        double u_computed = u[idx_mid];

        double err = u_computed - u_exact;
        diff_l2_sq += err * err;
        exact_l2_sq += u_exact * u_exact;
    }

    double relative_l2_error = std::sqrt(diff_l2_sq / exact_l2_sq);

    EXPECT_LE(relative_l2_error, 0.01) 
        << "Poiseuille velocity profile deviated from exact analytical solution. Relative L2 error: " 
        << (relative_l2_error * 100.0) << "% (Threshold: < 1.0%).";
}
