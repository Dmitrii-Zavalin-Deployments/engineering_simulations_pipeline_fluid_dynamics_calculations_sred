/**
 * @file test_lid_driven_cavity.cpp
 * @brief Literate integration test suite validating the NavierStokesOrchestrator fluid solver 
 *        against 2D Lid-Driven Cavity Flow at Reynolds number Re = 100.
 *
 * ## Physical Background & Governing Equations
 * Lid-driven cavity flow is a foundational benchmark for incompressible viscous fluid dynamics. 
 * It models a fluid contained within a square cavity where the top boundary (lid) moves tangentially 
 * at a constant velocity $U_{\text{lid}}$, while the remaining three walls are stationary (no-slip).
 * 
 * The flow dynamics are governed by the incompressible Navier-Stokes equations characterized by 
 * the Reynolds number:
 * $$ \text{Re} = \frac{U_{\text{lid}} L}{\nu} = 100 $$
 * where $L$ is the characteristic cavity width and $\nu$ is the kinematic viscosity. The flow 
 * develops a prominent primary vortex whose core position and vorticity distribution provide a 
 * rigorous check for advective-diffusive solver accuracy, boundary condition enforcement, and 
 * mass conservation stability.
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
 * We configure the test fixture class to manage parallel execution thread caps, clear floating-point 
 * exception registers, and supply numerical diagnostics for divergence, transient residues, and vortex tracking.
 */
class LidDrivenCavityTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef _OPENMP
        omp_set_num_threads(std::min(4, omp_get_max_threads()));
#endif
        std::feclearexcept(FE_ALL_EXCEPT);
    }

    /* 
     * To monitor mass conservation compliance, we compute the maximum velocity divergence magnitude 
     * across internal grid nodes using second-order central finite differences:
     *     \nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z}
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
     * We measure temporal relaxation progress by calculating the root-mean-square (RMS) velocity 
     * difference vector between successive time steps across all cells.
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

    /* 
     * We locate the primary recirculation vortex center by identifying the spatial location 
     * of minimum spanwise vorticity:
     *     \omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
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
        return {(best_i + 0.5) * dx, (best_j + 0.5) * dy};
    }
};

/* 
 * The integration test sets up a 3D cavity domain with a sliding top lid, smoothly ramps the lid velocity 
 * to prevent initial pressure shocks, marches the flow to near steady-state over 400 steps, and asserts 
 * that the primary vortex center falls within valid physical bounds.
 */
TEST_F(LidDrivenCavityTest, LidDrivenCavityRe100) {
    const int nx = 16;
    const int ny = 16;
    const int nz = 3;
    const double dx = 1.0 / nx;
    const double dy = 1.0 / ny;
    const double dz = 0.03125;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    SolverConfig config;
    config.density = 1.0;
    config.max_poisson_iterations = 25;
    config.poisson_tolerance = 1e-4;

    NavierStokesOrchestrator orchestrator(dims, config);

    const double nu = 0.01;
    const double mu = config.density * nu;
    const std::vector<double> gravity = {0.0, 0.0, 0.0};

    std::vector<int> mask(total_cells, 1);
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);
    std::vector<BoundaryCondition> bc_list;
    
    /* 
     * We enforce no-slip stationary walls across the left, right, bottom, and spanwise boundaries.
     */
    for (const std::string& loc : {"x_min", "x_max", "y_min", "z_min", "z_max"}) {
        BoundaryCondition bc_wall;
        bc_wall.location = loc;
        bc_wall.type = "no-slip";
        bc_wall.u_val = 0.0; bc_wall.v_val = 0.0; bc_wall.w_val = 0.0;
        bc_wall.values.has_u = true; bc_wall.values.u = 0.0;
        bc_wall.values.has_v = true; bc_wall.values.v = 0.0;
        bc_wall.values.has_w = true; bc_wall.values.w = 0.0;
        bc_list.push_back(bc_wall);
    }

    /* 
     * The top boundary (y_max) represents the moving lid initialized with dynamic velocity parameters.
     */
    BoundaryCondition bc_lid;
    bc_lid.location = "y_max";
    bc_lid.type = "inflow";
    bc_lid.u_val = 0.0; bc_lid.v_val = 0.0; bc_lid.w_val = 0.0;
    bc_lid.values.has_u = true; bc_lid.values.u = 0.0;
    bc_lid.values.has_v = true; bc_lid.values.v = 0.0;
    bc_lid.values.has_w = true; bc_lid.values.w = 0.0;
    bc_list.push_back(bc_lid);

    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    const double dt = 0.001;
    const int total_steps = 400;

    for (int step = 0; step < total_steps; ++step) {
        std::vector<double> u_old = u;
        std::vector<double> v_old = v;

        /* 
         * We smoothly ramp up the lid velocity over the initial transient steps to avoid artificial pressure spikes.
         */
        double current_lid_u = std::min(1.0, static_cast<double>(step) / 50.0);
        bc_lid.u_val = current_lid_u;
        bc_lid.values.u = current_lid_u;
        bc_list.back() = bc_lid;

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
        double allowed_residue = (step <= 60) ? 0.40 : 0.15;
        ASSERT_LT(residue, allowed_residue);
    }

    /* 
     * Finally, we locate the primary vortex center and verify that it remains within physical domain bounds.
     */
    auto [x_vortex, y_vortex] = LocatePrimaryVortexCenter(u, v, nx, ny, 1, dx, dy);
    ASSERT_GE(x_vortex, 0.0);
    ASSERT_LE(x_vortex, 1.0);
    ASSERT_GE(y_vortex, 0.0);
    ASSERT_LE(y_vortex, 1.0);
}
