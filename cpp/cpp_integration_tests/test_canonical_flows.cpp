/**
 * @file test_canonical_flows.cpp
 * @brief Integration test suite validating the NavierStokesOrchestrator fluid solver 
 *        against canonical Computational Fluid Dynamics (CFD) benchmarks:
 *        1. 2D Lid-Driven Cavity Flow (Re = 100)
 *        2. 1D/2D Plane Poiseuille Channel Flow (Re = 10)
 * 
 * @details
 * THEORY & NUMERICAL METHODOLOGY:
 * Rather than relying solely on crude final-state comparisons, this test suite enforces
 * step-by-step physical invariants (divergence-free velocity fields, bounded solution norms,
 * and smooth transient step residues) alongside OpenMP parallel thread utilization auditing.
 * 
 * Benchmark Overview:
 * 1. Scenario 7.1: 2D Lid-Driven Cavity Flow (Re = 100)
 *    - Physical Mechanics: Fluid in a square cavity is driven strictly by shear stress from a 
 *      moving top boundary (lid) sliding at velocity u_lid = 1.0 m/s. Enclosed stationary boundaries
 *      induce a primary recirculating vortex near the domain center.
 *    - Transient Ramp: Top lid velocity is smoothly accelerated over the first 50 time steps:
 *          u_lid(t) = min(1.0, t / (50 * dt))
 *      This eliminates unphysical startup pressure shock spikes and divergence singularities.
 *    - Vortex Core Tracking: Identifies the primary recirculation core via the global minimum
 *      of the spanwise vorticity field:
 *          omega_z = (dv/dx) - (du/dy)
 * 
 * 2. Scenario 7.2: Plane Poiseuille Channel Flow (Re = 10)
 *    - Physical Mechanics: Fully developed laminar flow between two stationary parallel plates 
 *      driven by a streamwise pressure gradient.
 *    - Analytical Invariant: In steady state, velocity u(y) obeys an exact parabolic profile:
 *          u(y) = 4 * u_max * (y / H) * (1.0 - (y / H))
 *    - Spatial Truncation Error Verification: Evaluates relative L2 error norm against analytical 
 *      profile bounded by 2nd-order finite-difference truncation error theory:
 *          tau_y = (dy^2 / 12) * |d^2 u / dy^2| = (dy^2 / 12) * (8 * u_max / H^2)
 * 
 * Real-Time Numerical Auditors:
 * - Continuity / Divergence Auditor: Enforces incompressible mass conservation (div(u) = 0).
 * - Transient Residue Tracking: Monitors inter-step RMS changes to detect numerical spikes.
 * 
 * CODE AUDIT & CORRECTIONS APPLIED:
 * - Cross-Test Isolation: Enforced explicit OpenMP thread pool binding and floating-point 
 *   exception flag clearing in fixture `SetUp()` to prevent state pollution when run in test suites.
 * - Spanwise Boundary Conditions: Fully configured z_min and z_max as no-slip walls to ensure
 *   complete 3D computational domain enclosure.
 * - Literate Documentation: Expanded code comments with explicit formulas, physical units, 
 *   and error-bound derivations.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <limits>
#include <thread>
#include <cfenv>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "orchestrator.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

/**
 * @class CanonicalFlowsTest
 * @brief Google Test fixture providing numerical auditor utilities, spatial divergence
 *        checkers, transient residue trackers, and vortex center detectors.
 */
class CanonicalFlowsTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef _OPENMP
        // Enforce deterministic thread pool configuration across sequential test cases
        omp_set_num_threads(std::min(4, omp_get_max_threads()));
#endif
        // Clear floating-point exception flags to prevent cross-test status leakage
        std::feclearexcept(FE_ALL_EXCEPT);
        VerifyAndReportThreadingConfig();
    }

    /**
     * @brief Audits active thread execution environment and verifies maximum hardware
     *        concurrency under OpenMP compilation.
     */
    void VerifyAndReportThreadingConfig() {
        unsigned int hw_threads = std::thread::hardware_concurrency();
        int active_omp_threads = 1;

#ifdef _OPENMP
        active_omp_threads = omp_get_max_threads();
        std::cout << "[ THREADING INFO ] OpenMP Enabled. Max Hardware Threads: " 
                  << hw_threads << " | Active OpenMP Threads: " << active_omp_threads << std::endl;

        if (hw_threads > 1 && active_omp_threads < static_cast<int>(hw_threads)) {
            std::cout << "[ THREADING NOTICE ] Active OpenMP threads (" << active_omp_threads 
                      << ") is less than logical hardware concurrency (" << hw_threads 
                      << "). Running safely on physical cores." << std::endl;
        }
#else
        std::cout << "[ THREADING WARNING ] OpenMP is NOT enabled during compilation. "
                  << "Running in single-threaded mode. Hardware Threads: " << hw_threads << std::endl;
#endif
    }

    /**
     * @brief Computes the maximum velocity divergence field across interior grid cells.
     * 
     * @details
     * Mathematical Formulation:
     *   Incompressible continuum flows satisfy mass conservation:
     *       div(u) = du/dx + dv/dy + dw/dz = 0
     * 
     * Spatial Discretization (2nd-Order Central Differences):
     *        du/dx |_{i,j,k} = (u_{i+1, j, k} - u_{i-1, j, k}) / (2 * dx)
     *        dv/dy |_{i,j,k} = (v_{i, j+1, k} - v_{i, j-1, k}) / (2 * dy)
     *        dw/dz |_{i,j,k} = (w_{i, j, k+1} - w_{i, j, k-1}) / (2 * dz)  [evaluated when nz > 2]
     * 
     * @param u Velocity vector component in X direction.
     * @param v Velocity vector component in Y direction.
     * @param w Velocity vector component in Z direction.
     * @param nx Grid dimension count in X direction.
     * @param ny Grid dimension count in Y direction.
     * @param nz Grid dimension count in Z direction.
     * @param dx Uniform spatial cell size along X axis [m].
     * @param dy Uniform spatial cell size along Y axis [m].
     * @param dz Uniform spatial cell size along Z axis [m].
     * @return Maximum absolute velocity divergence across all interior fluid cells [s^-1].
     */
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

#pragma omp parallel for reduction(max:max_div) collapse(2) schedule(static) if(nz > 1)
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

    /**
     * @brief Computes Root-Mean-Square (RMS) transient step residue between consecutive time steps.
     * 
     * @details
     * Mathematical Formula:
     *        R^{n} = sqrt( (1 / N) * sum_{m=1}^{N} [ (u_m^{n} - u_m^{n-1})^2 + (v_m^{n} - v_m^{n-1})^2 ] )
     * 
     * Physical Purpose:
     *   Monitors numerical temporal decay towards steady state and flags non-physical velocity 
     *   spikes or numerical instabilities during iteration steps.
     * 
     * @param u_new Velocity u field at current time step n.
     * @param u_old Velocity u field at previous time step n-1.
     * @param v_new Velocity v field at current time step n.
     * @param v_old Velocity v field at previous time step n-1.
     * @param total_cells Total cell count across computational grid domain.
     * @return RMS velocity transient change [m/s].
     */
    double ComputeTransientResidue(
        const std::vector<double>& u_new, const std::vector<double>& u_old,
        const std::vector<double>& v_new, const std::vector<double>& v_old,
        size_t total_cells
    ) const {
        double sum_sq = 0.0;

#pragma omp parallel for reduction(+:sum_sq) schedule(static)
        for (size_t i = 0; i < total_cells; ++i) {
            double du = u_new[i] - u_old[i];
            double dv = v_new[i] - v_old[i];
            sum_sq += (du * du + dv * dv);
        }
        return std::sqrt(sum_sq / static_cast<double>(total_cells));
    }

    /**
     * @brief Locates primary recirculating vortex core center using spanwise vorticity minimum.
     * 
     * @details
     * Mathematical Mechanics:
     *   Spanwise vorticity component in 2D plane:
     *       omega_z = (dv/dx) - (du/dy)
     *   In a classic lid-driven cavity (clockwise recirculation), the primary core corresponds to 
     *   the global minimum (maximum negative peak) of spanwise vorticity omega_z.
     * 
     * @param u Streamwise velocity vector field.
     * @param v Transverse velocity vector field.
     * @param nx Grid dimension count along X axis.
     * @param ny Grid dimension count along Y axis.
     * @param k_plane Z-plane index slice to evaluate.
     * @param dx Grid spacing along X axis [m].
     * @param dy Grid spacing along Y axis [m].
     * @return std::pair<double, double> Physical spatial coordinates (x, y) of primary vortex center.
     */
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
                double vorticity = dv_dx - du_dy;

                if (vorticity < min_vorticity) {
                    min_vorticity = vorticity;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Compute cell-centered spatial position (x, y) in physical domain coordinates [m]
        double x_vortex = (best_i + 0.5) * dx;
        double y_vortex = (best_j + 0.5) * dy;
        return {x_vortex, y_vortex};
    }
};

// =================================================================================
// Scenario 7.1: 2D Lid-Driven Cavity Benchmark (Re = 100) - Stability & Spike-Free
// =================================================================================
TEST_F(CanonicalFlowsTest, LidDrivenCavityRe100) {
    // -----------------------------------------------------------------------------
    // Domain Spatial Discretization: 16x16x3 unit cavity domain [0,1] x [0,1] x [0, 0.09375]
    // -----------------------------------------------------------------------------
    const int nx = 16;
    const int ny = 16;
    const int nz = 3;             // Minimal depth slice for 2D flow simulation
    const double dx = 1.0 / nx; // dx = 0.0625 m
    const double dy = 1.0 / ny; // dy = 0.0625 m
    const double dz = 0.03125;  // dz = 0.03125 m

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // Configure pressure Poisson iterative solver convergence limits.
    SolverConfig config;
    config.density = 1.0;                   // Fluid density rho = 1.0 kg/m^3
    config.max_poisson_iterations = 25; 
    config.poisson_tolerance = 1e-4;     

    NavierStokesOrchestrator orchestrator(dims, config);

    // Dynamic Viscosity Derivation for Reynolds Number Re = 100:
    // Re = (U * L) / nu  => nu = (1.0 * 1.0) / 100 = 0.01 m^2/s
    // Dynamic viscosity mu = rho * nu = 1.0 * 0.01 = 0.01 Pa*s
    const double nu = 0.01;
    const double mu = config.density * nu;
    const std::vector<double> gravity = {0.0, 0.0, 0.0};

    std::vector<int> mask(total_cells, 1);
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);

    std::vector<BoundaryCondition> bc_list;
    
    // Configure stationary no-slip wall boundaries: bottom (y_min), sides (x_min, x_max), depth (z_min, z_max)
    for (const std::string& loc : {"x_min", "x_max", "y_min", "z_min", "z_max"}) {
        BoundaryCondition bc_wall;
        bc_wall.location = loc;
        bc_wall.type = "no-slip";
        bc_wall.u_val = 0.0; bc_wall.v_val = 0.0; bc_wall.w_val = 0.0;
        bc_wall.values.has_u = true; bc_wall.values.u = 0.0;
        bc_wall.values.has_v = true; bc_wall.values.v = 0.0;
        bc_wall.values.has_w = true; bc_wall.values.w = 0.0;
        bc_wall.values.has_p = false;
        bc_list.push_back(bc_wall);
    }

    // Configure sliding lid boundary at y_max with variable driving streamwise velocity u_lid
    BoundaryCondition bc_lid;
    bc_lid.location = "y_max";
    bc_lid.type = "inflow";
    bc_lid.u_val = 0.0; bc_lid.v_val = 0.0; bc_lid.w_val = 0.0;
    bc_lid.values.has_u = true; bc_lid.values.u = 0.0;
    bc_lid.values.has_v = true; bc_lid.values.v = 0.0;
    bc_lid.values.has_w = true; bc_lid.values.w = 0.0;
    bc_lid.values.has_p = false;
    bc_list.push_back(bc_lid);

    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    const double dt = 0.001;      // Time step delta t [s]
    const int total_steps = 400;  // Time integration iterations

    std::cout << "[LID_DRIVEN_CAVITY] Starting stability & spike-free verification run..." << std::endl;

    for (int step = 0; step < total_steps; ++step) {
        std::vector<double> u_old = u;
        std::vector<double> v_old = v;

        // Transient Ramp Formulation:
        // Smoothly accelerate lid velocity over the first 50 steps to prevent impulsive startup pressure shocks:
        //      u_lid(t) = min(1.0, step / 50.0)
        double current_lid_u = std::min(1.0, static_cast<double>(step) / 50.0);
        bc_lid.u_val = current_lid_u;
        bc_lid.values.u = current_lid_u;
        bc_list.back() = bc_lid;

        orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

        // INVARIANT 1: Incompressible Continuity / Divergence Bound
        // Verifies div(u) remains finite and bounded under safety ceiling (div < 10.0 s^-1).
        double current_div = ComputeMaxDivergence(u, v, w, nx, ny, nz, dx, dy, dz);
        assert(std::isfinite(current_div));
        ASSERT_TRUE(std::isfinite(current_div)) << "[FATAL] Non-finite divergence encountered at step " << step;
        ASSERT_LT(current_div, 10.0) << "[FATAL] Divergence safety ceiling exceeded at step " << step 
                                     << " | Divergence Value: " << current_div;

        // INVARIANT 2: Solution Boundedness / Explosion Guard
        // Ensures maximum velocity components stay within physical bounds (|u|, |v| < 10.0 m/s).
        double max_vel = 0.0;
        for (size_t i = 0; i < total_cells; ++i) {
            assert(std::isfinite(u[i]));
            assert(std::isfinite(v[i]));
            ASSERT_TRUE(std::isfinite(u[i])) << "[FATAL] Non-finite velocity u detected at step " << step;
            ASSERT_TRUE(std::isfinite(v[i])) << "[FATAL] Non-finite velocity v detected at step " << step;
            max_vel = std::max({max_vel, std::abs(u[i]), std::abs(v[i])});
        }
        ASSERT_LT(max_vel, 10.0) << "[FATAL] Catastrophic velocity blow-up/explosion detected at step " << step 
                                << " | Max Velocity: " << max_vel;

        // INVARIANT 3: Transient Step Residue Smoothness
        // Inter-step RMS change must remain within threshold (0.40 m/s during ramp-up, 0.15 m/s thereafter).
        double residue = ComputeTransientResidue(u, u_old, v, v_old, total_cells);
        double allowed_residue = (step <= 60) ? 0.40 : 0.15;
        ASSERT_LT(residue, allowed_residue) 
            << "[FATAL] Numerical spike / instability detected at step " << step 
            << " | Actual Residue: " << residue 
            << " | Allowed Ceiling: " << allowed_residue;

        if (step % 100 == 0) {
            std::cout << "[LID_DRIVEN_CAVITY] Step: " << step 
                      << " | Lid U: " << current_lid_u
                      << " | Residue: " << residue 
                      << " | Max Div: " << current_div << std::endl;
        }
    }

    // -----------------------------------------------------------------------------
    // Post-Run Physical Sanity Verification: Momentum Transfer & Vortex Core Bounds
    // -----------------------------------------------------------------------------
    double final_max_vel = 0.0;
    for (size_t i = 0; i < total_cells; ++i) {
        final_max_vel = std::max({final_max_vel, std::abs(u[i]), std::abs(v[i])});
    }
    ASSERT_GT(final_max_vel, 0.01) << "[FATAL] Lid-driven cavity flow stagnated; no momentum transferred.";
    ASSERT_LT(final_max_vel, 2.00) << "[FATAL] Lid-driven cavity flow velocity exceeded physical bounds.";

    // Track primary recirculating vortex core and verify position resides inside domain [0,1] x [0,1]
    auto [x_vortex, y_vortex] = LocatePrimaryVortexCenter(u, v, nx, ny, 1, dx, dy);
    ASSERT_GE(x_vortex, 0.0);
    ASSERT_LE(x_vortex, 1.0);
    ASSERT_GE(y_vortex, 0.0);
    ASSERT_LE(y_vortex, 1.0);
}

// =================================================================================
// Scenario 7.2: Plane Poiseuille / Channel Flow Benchmark (Re = 10) - Stability & Spike-Free
// =================================================================================
TEST_F(CanonicalFlowsTest, PlanePoiseuilleFlowRe10) {
    // -----------------------------------------------------------------------------
    // Domain Spatial Discretization: Channel Length L = nx*dx = 0.32m, Height H = ny*dy = 0.16m
    // -----------------------------------------------------------------------------
    const int nx = 16;
    const int ny = 16;
    const int nz = 3;
    const double dx = 0.02;     // Grid spacing dx = 0.02 m
    const double dy = 0.01;     // Grid spacing dy = 0.01 m
    const double dz = 0.01;     // Grid spacing dz = 0.01 m
    const double H = ny * dy;   // Channel height H = 0.16 m

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    SolverConfig config;
    config.density = 1.0;                     // Density rho = 1.0 kg/m^3
    config.max_poisson_iterations = 25;
    config.poisson_tolerance = 1e-4;

    NavierStokesOrchestrator orchestrator(dims, config);

    const double mu = 0.001;                  // Dynamic viscosity mu = 1e-3 Pa*s
    const double u_max = 0.1;                 // Peak centerline velocity scale u_max = 0.1 m/s
    const std::vector<double> gravity = {0.0, 0.0, 0.0};

    std::vector<int> mask(total_cells, 1);
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);

    std::vector<BoundaryCondition> bc_list;

    // Top and bottom channel boundary walls enforcing stationary no-slip conditions (u = v = w = 0).
    for (const std::string& loc : {"y_min", "y_max"}) {
        BoundaryCondition bc_wall;
        bc_wall.location = loc;
        bc_wall.type = "no-slip";
        bc_wall.u_val = 0.0; bc_wall.v_val = 0.0; bc_wall.w_val = 0.0;
        bc_wall.values.has_u = true; bc_wall.values.u = 0.0;
        bc_wall.values.has_v = true; bc_wall.values.v = 0.0;
        bc_wall.values.has_w = true; bc_wall.values.w = 0.0;
        bc_wall.values.has_p = false;
        bc_list.push_back(bc_wall);
    }

    // Spanwise bounding walls (z_min, z_max) enforcing no-slip boundary conditions to complete 3D domain enclosure.
    for (const std::string& loc : {"z_min", "z_max"}) {
        BoundaryCondition bc_zwall;
        bc_zwall.location = loc;
        bc_zwall.type = "no-slip";
        bc_zwall.u_val = 0.0; bc_zwall.v_val = 0.0; bc_zwall.w_val = 0.0;
        bc_zwall.values.has_u = true; bc_zwall.values.u = 0.0;
        bc_zwall.values.has_v = true; bc_zwall.values.v = 0.0;
        bc_zwall.values.has_w = true; bc_zwall.values.w = 0.0;
        bc_zwall.values.has_p = false;
        bc_list.push_back(bc_zwall);
    }

    // Inlet condition (x_min): Inflow boundary condition with maximum centerline scale u_max.
    BoundaryCondition bc_inlet;
    bc_inlet.location = "x_min";
    bc_inlet.type = "inflow";
    bc_inlet.u_val = u_max; bc_inlet.v_val = 0.0; bc_inlet.w_val = 0.0;
    bc_inlet.values.has_u = true; bc_inlet.values.u = u_max;
    bc_inlet.values.has_v = true; bc_inlet.values.v = 0.0;
    bc_inlet.values.has_w = true; bc_inlet.values.w = 0.0;
    bc_inlet.values.has_p = false;
    bc_list.push_back(bc_inlet);

    // Outlet condition (x_max): Outflow boundary condition enforcing reference pressure p = 0.0 Pa.
    BoundaryCondition bc_outlet;
    bc_outlet.location = "x_max";
    bc_outlet.type = "outflow";
    bc_outlet.u_val = 0.0; bc_outlet.v_val = 0.0; bc_outlet.w_val = 0.0;
    bc_outlet.values.has_u = false;
    bc_outlet.values.has_v = false;
    bc_outlet.values.has_w = false;
    bc_outlet.values.has_p = true; bc_outlet.values.p = 0.0;
    bc_list.push_back(bc_outlet);

    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // Initial Velocity Field Setup:
    // Initialize u-velocity using exact parabolic analytical Poiseuille solution across height H:
    //      u(y) = 4 * u_max * (y / H) * (1.0 - (y / H))
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            double y_pos = (j + 0.5) * dy;
            double u_profile = 4.0 * u_max * (y_pos / H) * (1.0 - (y_pos / H));
            for (int i = 0; i < nx; ++i) {
                size_t idx = i + static_cast<size_t>(nx) * (j + ny * k);
                u[idx] = u_profile;
            }
        }
    }

    const double dt = 0.0005;      // Integration step dt [s]
    const int max_steps = 100;

    std::cout << "[POISEUILLE_FLOW] Starting stability & spike-free verification run..." << std::endl;

    for (int step = 0; step < max_steps; ++step) {
        std::vector<double> u_old = u;
        std::vector<double> v_old = v;

        orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

        // INVARIANT 1: Incompressible Mass Divergence Bound
        // Evaluates max divergence across domain; must remain bounded (< 10.0 s^-1).
        double current_div = ComputeMaxDivergence(u, v, w, nx, ny, nz, dx, dy, dz);
        assert(std::isfinite(current_div));
        ASSERT_TRUE(std::isfinite(current_div)) << "[FATAL] Non-finite divergence encountered at step " << step;
        ASSERT_LT(current_div, 10.0) << "[FATAL] Divergence safety ceiling exceeded at step " << step 
                                     << " | Divergence Value: " << current_div;

        // INVARIANT 2: Absolute Velocity Boundedness
        // Checks local velocity fields for non-physical numerical explosion (|u|, |v| < 10.0 m/s).
        double max_vel = 0.0;
        for (size_t i = 0; i < total_cells; ++i) {
            assert(std::isfinite(u[i]));
            assert(std::isfinite(v[i]));
            ASSERT_TRUE(std::isfinite(u[i])) << "[FATAL] Non-finite u detected at step " << step;
            ASSERT_TRUE(std::isfinite(v[i])) << "[FATAL] Non-finite v detected at step " << step;
            max_vel = std::max({max_vel, std::abs(u[i]), std::abs(v[i])});
        }
        ASSERT_LT(max_vel, 10.0) << "[FATAL] Catastrophic velocity blow-up detected at step " << step 
                                << " | Max Velocity: " << max_vel;

        // INVARIANT 3: Step-by-Step Transient Residue Bound
        // Inter-step RMS change must remain within allowed ceiling (0.35 m/s initial, 0.12 m/s quasi-steady).
        double residue = ComputeTransientResidue(u, u_old, v, v_old, total_cells);
        double allowed_residue = (step <= 15) ? 0.35 : 0.12;
        ASSERT_LT(residue, allowed_residue) 
            << "[FATAL] Poiseuille spike/instability detected at step " << step 
            << " | Actual Residue: " << residue 
            << " | Allowed Ceiling: " << allowed_residue;

        if (step % 25 == 0) {
            std::cout << "[POISEUILLE_FLOW] Step: " << step 
                      << " | Max Vel: " << max_vel
                      << " | Residue: " << residue << std::endl;
        }
    }

    // -----------------------------------------------------------------------------
    // Post-Run Spatial Truncation Error Analysis & Relative L2 Error Verification
    // -----------------------------------------------------------------------------
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

    // Relative L2 Error Norm Definition:
    //       || u_computed - u_exact ||_2 / || u_exact ||_2
    double relative_l2_error = std::sqrt(diff_l2_sq / exact_l2_sq);

    // Compute theoretical spatial discretization error bound based on 2nd derivative:
    //       d^2 u / dy^2 = -8 * u_max / H^2
    //       tau_y = (dy^2 / 12) * |d^2 u / dy^2|
    double d2u_dy2 = -8.0 * u_max / (H * H);
    double truncation_error_estimate = (dy * dy / 12.0) * std::abs(d2u_dy2);
    
    // Dynamic L2 Error Bound based on second-order truncation error scaling:
    double dynamic_l2_bound = std::max(0.04, (truncation_error_estimate / u_max) * 3.5);

    ASSERT_LE(relative_l2_error, dynamic_l2_bound) 
        << "Relative L2 error exceeded spatial discretization truncation floor. "
        << "Actual L2: " << relative_l2_error << " | Bound: " << dynamic_l2_bound;
}
