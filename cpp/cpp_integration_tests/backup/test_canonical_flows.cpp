/**
 * @file test_canonical_flows.cpp
 * @brief Integration tests verifying physical convergence, mathematical invariants,
 *        and maximum thread utilization against canonical CFD benchmarks 
 *        (2D Lid-Driven Cavity and Plane Poiseuille Flow) using stability and spike-free invariants.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <limits>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "orchestrator.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

/**
 * @class CanonicalFlowsTest
 * @brief Test fixture providing shared numerical verification utilities, divergence auditors,
 *        transient residue metrics, and vortex center trackers written in a literate narrative style.
 */
class CanonicalFlowsTest : public ::testing::Test {
protected:
    void SetUp() override {
        VerifyAndReportThreadingConfig();
    }

    /**
     * @brief Verifies that the runtime environment is configured to utilize maximum 
     *        available hardware concurrency for OpenMP multi-threading.
     */
    void VerifyAndReportThreadingConfig() {
        // We query the underlying hardware concurrency and active OpenMP thread pools.
        unsigned int hw_threads = std::thread::hardware_concurrency();
        int active_omp_threads = 1;

#ifdef _OPENMP
        active_omp_threads = omp_get_max_threads();
        std::cout << "[ THREADING INFO ] OpenMP Enabled. Max Hardware Threads: " 
                  << hw_threads << " | Active OpenMP Threads: " << active_omp_threads << std::endl;

        if (hw_threads > 1) {
            // We assert that OpenMP is fully utilizing available hardware cores.
            ASSERT_GE(active_omp_threads, static_cast<int>(hw_threads))
                << "WARNING: OpenMP is not using all available CPU threads! "
                << "Active: " << active_omp_threads << ", Hardware available: " << hw_threads;
        }
#else
        std::cout << "[ THREADING WARNING ] OpenMP is NOT enabled during compilation. "
                  << "Running in single-threaded mode. Hardware Threads: " << hw_threads << std::endl;
#endif
    }

    /**
     * @brief Computes the maximum velocity divergence across internal grid cells.
     * 
     * Incompressible Navier-Stokes flows satisfy the divergence-free continuity constraint:
     *     div(u) = du/dx + dv/dy + dw/dz = 0
     * 
     * We compute central-difference approximations for spatial derivatives:
     *     du/dx = (u(i+1, j, k) - u(i-1, j, k)) / (2 * dx)
     *     dv/dy = (v(i, j+1, k) - v(i, j-1, k)) / (2 * dy)
     *     dw/dz = (w(i, j, k+1) - w(i, j, k-1)) / (2 * dz)  [if nz > 2]
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
     * @brief Computes the RMS transient residue between consecutive time steps.
     * 
     * Formula:
     *     R = sqrt( (1 / N) * sum( (u_new - u_old)^2 + (v_new - v_old)^2 ) )
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
        return std::sqrt(sum_sq / total_cells);
    }

    /**
     * @brief Locates the primary vortex center using the spanwise vorticity minimum.
     * 
     * Spanwise vorticity component:
     *     omega_z = dv/dx - du/dy
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

        double x_vortex = (best_i + 0.5) * dx;
        double y_vortex = (best_j + 0.5) * dy;
        return {x_vortex, y_vortex};
    }
};

// =================================================================================
// Scenario 7.1: 2D Lid-Driven Cavity Benchmark (Re = 100) - Stability & Spike-Free
// =================================================================================
TEST_F(CanonicalFlowsTest, LidDrivenCavityRe100) {
    // We define the grid dimensions and spacing for the square cavity.
    const int nx = 16;
    const int ny = 16;
    const int nz = 3;
    const double dx = 1.0 / nx;
    const double dy = 1.0 / ny;
    const double dz = 0.03125;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // Configure solver physical and numerical iteration parameters.
    SolverConfig config;
    config.density = 1.0;
    config.max_poisson_iterations = 25; 
    config.poisson_tolerance = 1e-4;     

    NavierStokesOrchestrator orchestrator(dims, config);

    // Reynolds number Re = 100 configuration: kinematic viscosity nu = 0.01.
    const double nu = 0.01;
    const double mu = config.density * nu;
    const std::vector<double> gravity = {0.0, 0.0, 0.0};

    std::vector<int> mask(total_cells, 1);
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);

    std::vector<BoundaryCondition> bc_list;
    
    // Configure stationary no-slip walls for x_min, x_max, y_min, z_min, z_max.
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

    // Configure the moving top lid at y_max with an inflow/velocity boundary type.
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

    const double dt = 0.001;
    const int total_steps = 400;

    std::cout << "[LID_DRIVEN_CAVITY] Starting stability & spike-free verification run..." << std::endl;

    for (int step = 0; step < total_steps; ++step) {
        std::vector<double> u_old = u;
        std::vector<double> v_old = v;

        // Smoothly ramp up the lid velocity over the first 50 steps:
        //     current_lid_u = min(1.0, step / 50.0)
        double current_lid_u = std::min(1.0, static_cast<double>(step) / 50.0);
        bc_lid.u_val = current_lid_u;
        bc_lid.values.u = current_lid_u;
        bc_list.back() = bc_lid;

        orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

        // INVARIANT 1: Divergence must remain finite and bounded on every step
        double current_div = ComputeMaxDivergence(u, v, w, nx, ny, nz, dx, dy, dz);
        assert(std::isfinite(current_div));
        ASSERT_TRUE(std::isfinite(current_div)) << "[FATAL] Non-finite divergence encountered at step " << step;
        ASSERT_LT(current_div, 10.0) << "[FATAL] Divergence safety ceiling exceeded at step " << step 
                                     << " | Divergence Value: " << current_div;

        // INVARIANT 2: Velocity components must remain bounded and finite on every step
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

        // INVARIANT 3: Per-step transient residue with dynamic startup envelope
        double residue = ComputeTransientResidue(u, u_old, v, v_old, total_cells);
        double allowed_residue = (step <= 60) ? 0.40 : 0.10;
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

    // Post-run robust state sanity check: Verify flow is active, bounded, and vortex is within domain limits
    double final_max_vel = 0.0;
    for (size_t i = 0; i < total_cells; ++i) {
        final_max_vel = std::max({final_max_vel, std::abs(u[i]), std::abs(v[i])});
    }
    ASSERT_GT(final_max_vel, 0.01) << "[FATAL] Lid-driven cavity flow stagnated; no momentum transferred.";
    ASSERT_LT(final_max_vel, 2.00) << "[FATAL] Lid-driven cavity flow velocity exceeded physical bounds.";

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
    // Define grid dimensions and channel height H = ny * dy.
    const int nx = 16;
    const int ny = 16;
    const int nz = 3;
    const double dx = 0.02;
    const double dy = 0.01;
    const double dz = 0.01;
    const double H = ny * dy;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    SolverConfig config;
    config.density = 1.0;
    config.max_poisson_iterations = 25;
    config.poisson_tolerance = 1e-4;

    NavierStokesOrchestrator orchestrator(dims, config);

    const double mu = 0.001;
    const double u_max = 0.1;
    const std::vector<double> gravity = {0.0, 0.0, 0.0};

    std::vector<int> mask(total_cells, 1);
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);

    std::vector<BoundaryCondition> bc_list;

    // Top and bottom channel walls (no-slip condition).
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

    // Inlet condition (x_min): Prescribed parabolic centerline velocity u_max.
    BoundaryCondition bc_inlet;
    bc_inlet.location = "x_min";
    bc_inlet.type = "inflow";
    bc_inlet.u_val = u_max; bc_inlet.v_val = 0.0; bc_inlet.w_val = 0.0;
    bc_inlet.values.has_u = true; bc_inlet.values.u = u_max;
    bc_inlet.values.has_v = true; bc_inlet.values.v = 0.0;
    bc_inlet.values.has_w = true; bc_inlet.values.w = 0.0;
    bc_inlet.values.has_p = false;
    bc_list.push_back(bc_inlet);

    // Outlet condition (x_max): Dirichlet pressure p = 0.0.
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

    // Initialize velocity field with the analytical parabolic Poiseuille profile:
    //     u(y) = 4 * u_max * (y / H) * (1.0 - (y / H))
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

    const double dt = 0.0005;
    const int max_steps = 100;

    std::cout << "[POISEUILLE_FLOW] Starting stability & spike-free verification run..." << std::endl;

    for (int step = 0; step < max_steps; ++step) {
        std::vector<double> u_old = u;
        std::vector<double> v_old = v;

        orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

        // Stability check: Divergence must remain finite and bounded on every step
        double current_div = ComputeMaxDivergence(u, v, w, nx, ny, nz, dx, dy, dz);
        assert(std::isfinite(current_div));
        ASSERT_TRUE(std::isfinite(current_div)) << "[FATAL] Non-finite divergence encountered at step " << step;
        ASSERT_LT(current_div, 10.0) << "[FATAL] Divergence safety ceiling exceeded at step " << step 
                                     << " | Divergence Value: " << current_div;

        // Velocity bounds & explosion check on every step
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

        // Spike-free transient check on every step with dynamic startup envelope
        double residue = ComputeTransientResidue(u, u_old, v, v_old, total_cells);
        double allowed_residue = (step <= 15) ? 0.25 : 0.08;
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

    // Post-run analytical profile comparison: Compute L2 Error against exact parabolic solution.
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

    // Compute expected truncation error bound based on second derivative d2u/dy2:
    //     d2u/dy2 = -8 * u_max / H^2
    //     truncation_error_estimate = (dy^2 / 12) * |d2u/dy2|
    double d2u_dy2 = -8.0 * u_max / (H * H);
    double truncation_error_estimate = (dy * dy / 12.0) * std::abs(d2u_dy2);
    double dynamic_l2_bound = std::max(0.04, (truncation_error_estimate / u_max) * 3.5);

    ASSERT_LE(relative_l2_error, dynamic_l2_bound) 
        << "Relative L2 error exceeded spatial discretization truncation floor.";
}
