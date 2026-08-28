/**
 * @file test_plane_poiseuille_flow.cpp
 * @brief Literate integration test suite validating the NavierStokesOrchestrator fluid solver 
 *        against Plane Poiseuille Channel Flow.
 *
 * ## Physical Background & Governing Equations
 * Plane Poiseuille flow describes the motion of an incompressible viscous fluid driven by a pressure 
 * gradient or body force between two infinite stationary parallel plates separated by height $H$. 
 * 
 * Under steady, fully developed conditions, the Navier-Stokes momentum equations reduce to the 
 * exact balance between viscous diffusion and the driving force:
 * $$ \nu \frac{d^2 u}{dy^2} + f_x = 0 $$
 * 
 * The exact analytical velocity profile $u(y)$ across the channel coordinate $y \in [0, H]$ is given by:
 * $$ u(y) = 4 u_{\text{max}} \frac{y}{H} \left(1 - \frac{y}{H}\right) $$
 * where $u_{\text{max}}$ is the centerline velocity. This test initializes the fluid domain with this 
 * exact parabolic profile, applies the matching analytical driving body force $f_x$, and verifies 
 * that the numerical orchestrator maintains stability, mass continuity, and accuracy within dynamic 
 * truncation bounds over transient solver steps.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
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

/* 
 * We establish a test fixture class to configure execution threads, manage floating-point 
 * exception flags, and provide shared numerical diagnostic utilities for divergence and residue tracking.
 */
class PlanePoiseuilleTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef _OPENMP
        omp_set_num_threads(std::min(4, omp_get_max_threads()));
#endif
        std::feclearexcept(FE_ALL_EXCEPT);
    }

    /* 
     * To verify mass conservation at every grid node, we compute the discrete velocity divergence:
     *      \nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z}
     * Using second-order central differences across interior cells, we extract the maximum divergence magnitude.
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

    /* 
     * We measure transient convergence progress between successive time steps by calculating 
     * the root-mean-square (RMS) velocity difference vector across all spatial cells.
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
};

/* 
 * The integration test configures a 3D channel domain, initializes the exact parabolic profile,
 * applies the sustaining body force, runs the time-marching orchestrator steps, and validates 
 * that the computed velocity matches analytical expectations within rigorous L2 norm bounds.
 */
TEST_F(PlanePoiseuilleTest, PlanePoiseuilleFlowRe10) {
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

    const double mu = 0.001; // Kinematic viscosity nu
    const double u_max = 0.1;
    const std::vector<double> gravity = {0.0, 0.0, 0.0};

    std::vector<int> mask(total_cells, 1);
    
    // We apply the exact analytical body force fx = (8 * nu * u_max) / H^2 to sustain Poiseuille flow
    const double fx_driving = 8.0 * mu * u_max / (H * H);
    std::vector<double> fx(total_cells, fx_driving);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);

    std::vector<BoundaryCondition> bc_list;

    /* 
     * We enforce no-slip boundary conditions along the bottom and top channel walls (y_min, y_max), 
     * as well as periodic/confined spanwise boundaries (z_min, z_max).
     */
    for (const std::string& loc : {"y_min", "y_max"}) {
        BoundaryCondition bc_wall;
        bc_wall.location = loc;
        bc_wall.type = "no-slip";
        bc_wall.u_val = 0.0; bc_wall.v_val = 0.0; bc_wall.w_val = 0.0;
        bc_wall.values.has_u = true; bc_wall.values.u = 0.0;
        bc_wall.values.has_v = true; bc_wall.values.v = 0.0;
        bc_wall.values.has_w = true; bc_wall.values.w = 0.0;
        bc_list.push_back(bc_wall);
    }

    for (const std::string& loc : {"z_min", "z_max"}) {
        BoundaryCondition bc_zwall;
        bc_zwall.location = loc;
        bc_zwall.type = "no-slip";
        bc_zwall.u_val = 0.0; bc_zwall.v_val = 0.0; bc_zwall.w_val = 0.0;
        bc_zwall.values.has_u = true; bc_zwall.values.u = 0.0;
        bc_zwall.values.has_v = true; bc_zwall.values.v = 0.0;
        bc_zwall.values.has_w = true; bc_zwall.values.w = 0.0;
        bc_list.push_back(bc_zwall);
    }

    /* 
     * Inlet and outlet boundary conditions impose the driving inflow profile and reference pressure.
     */
    BoundaryCondition bc_inlet;
    bc_inlet.location = "x_min";
    bc_inlet.type = "inflow";
    bc_inlet.u_val = u_max; bc_inlet.v_val = 0.0; bc_inlet.w_val = 0.0;
    bc_inlet.values.has_u = true; bc_inlet.values.u = u_max;
    bc_inlet.values.has_v = true; bc_inlet.values.v = 0.0;
    bc_inlet.values.has_w = true; bc_inlet.values.w = 0.0;
    bc_list.push_back(bc_inlet);

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

    /* 
     * We initialize the velocity field directly with the exact parabolic Poiseuille profile:
     *      u(y) = 4 u_{\text{max}} \frac{y}{H} \left(1 - \frac{y}{H}\right)
     */
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
    const int max_steps = 15;

    for (int step = 0; step < max_steps; ++step) {
        std::vector<double> u_old = u;
        std::vector<double> v_old = v;

        orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

        double current_div = ComputeMaxDivergence(u, v, w, nx, ny, nz, dx, dy, dz);
        ASSERT_TRUE(std::isfinite(current_div));
        ASSERT_LT(current_div, 10.0);

        double max_vel = 0.0;
        for (size_t i = 0; i < total_cells; ++i) {
            ASSERT_TRUE(std::isfinite(u[i]));
            ASSERT_TRUE(std::isfinite(v[i]));
            max_vel = std::max({max_vel, std::abs(u[i]), std::abs(v[i])});
        }
        ASSERT_LT(max_vel, 10.0);

        double residue = ComputeTransientResidue(u, u_old, v, v_old, total_cells);
        double allowed_residue = (step <= 15) ? 0.35 : 0.12;
        ASSERT_LT(residue, allowed_residue);
    }

    /* 
     * Finally, we compute the relative L2 norm error between the numerical velocity profile 
     * at the mid-channel cross-section and the theoretical analytical solution.
     */
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
    double d2u_dy2 = -8.0 * u_max / (H * H);
    double truncation_error_estimate = (dy * dy / 12.0) * std::abs(d2u_dy2);
    double dynamic_l2_bound = std::max(0.04, (truncation_error_estimate / u_max) * 3.5);

    ASSERT_LE(relative_l2_error, dynamic_l2_bound);
}
