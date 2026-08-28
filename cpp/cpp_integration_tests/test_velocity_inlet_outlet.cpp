/**
 * @file test_velocity_inlet_outlet.cpp
 * @brief Scenario 3.1: Velocity Inlet / Pressure Outlet Boundary Verification
 *        Refactored to use NavierStokesOrchestrator and stability invariants.
 * 
 * LITERATE TESTING NARRATIVE & MATHEMATICAL FORMULATION:
 * ---------------------------------------------------------------------------------
 * This integration test verifies the robust coupling of velocity inlet and pressure 
 * outlet boundary conditions in the Navier-Stokes orchestrator.
 * 
 * Grid spacing increments are defined as:
 *      dx = (x_max - x_min) / nx
 *      dy = (y_max - y_min) / ny
 *      dz = (z_max - z_min) / nz
 * 
 * Mass conservation across inlet and outlet faces requires:
 *      m_dot_inlet  = sum(rho * u_inlet * dy * dz)
 *      m_dot_outlet = sum(rho * u_outlet * dy * dz)
 *      m_dot_inlet == m_dot_outlet
 * 
 * Divergence constraint:
 *      div(u) = du/dx + dv/dy + dw/dz == 0
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cassert>

#include "orchestrator.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

// We define a helper function to compute the maximum velocity divergence across the interior cells:
//      div = |du/dx + dv/dy + dw/dz|
static double ComputeMaxDivergence(
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    int nx, int ny, int nz,
    double dx, double dy, double dz
) {
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

TEST(BoundaryConditionsTest, VelocityInletPressureOutlet) {
    // We define grid dimensions and spatial steps:
    const int nx = 10;
    const int ny = 8;
    const int nz = 8;
    const double dx = 0.1;
    const double dy = 0.1;
    const double dz = 0.1;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    assert(nx > 0 && ny > 0 && nz > 0);
    assert(dx > 0.0 && dy > 0.0 && dz > 0.0);

    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // We define fluid properties and simulation parameters:
    const double density = 1000.0;
    const double mu = 0.001;
    const double nu = mu / density;
    const double dt = 0.001;
    const double U_0 = 1.0;
    const int total_steps = 200; // Optimized step count for rapid integration test execution

    assert(density > 0.0);
    assert(nu >= 0.0);
    assert(dt > 0.0);
    assert(total_steps > 0);

    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = 50; 
    config.poisson_tolerance = 1e-5;     

    NavierStokesOrchestrator orchestrator(dims, config);

    // We initialize field vectors:
    std::vector<int> mask(total_cells, 1);
    std::vector<double> u(total_cells, U_0); // Initialize domain with U_0 to establish baseline mass flux
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);
    std::vector<double> gravity = {0.0, 0.0, 0.0};

    std::vector<BoundaryCondition> bc_list;

    // We configure the velocity inlet at x_min:
    BoundaryCondition bc_inlet;
    bc_inlet.location = "x_min";
    bc_inlet.type = "velocity_inlet";
    bc_inlet.u_val = U_0; bc_inlet.v_val = 0.0; bc_inlet.w_val = 0.0;
    bc_inlet.values.has_u = true; bc_inlet.values.u = U_0;
    bc_inlet.values.has_v = true; bc_inlet.values.v = 0.0;
    bc_inlet.values.has_w = true; bc_inlet.values.w = 0.0;
    bc_inlet.values.has_p = false;
    bc_list.push_back(bc_inlet);

    // We configure the pressure outlet at x_max:
    BoundaryCondition bc_outlet;
    bc_outlet.location = "x_max";
    bc_outlet.type = "pressure_outlet";
    bc_outlet.scalar_p = 0.0;
    bc_outlet.values.has_u = false;
    bc_outlet.values.has_v = false;
    bc_outlet.values.has_w = false;
    bc_outlet.values.has_p = true; bc_outlet.values.p = 0.0;
    bc_list.push_back(bc_outlet);
    assert(bc_list.size() == 2);

    std::cout << "[VELOCITY_INLET_OUTLET] Starting orchestrated verification run..." << std::endl;

    // We execute the simulation steps and verify divergence stability invariants at each step:
    for (int step = 0; step < total_steps; ++step) {
        orchestrator.step(dt, nu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

        // Stability check: Divergence safety check on every step
        double current_div = ComputeMaxDivergence(u, v, w, nx, ny, nz, dx, dy, dz);
        assert(std::isfinite(current_div));
        assert(current_div < 10.0);

        ASSERT_TRUE(std::isfinite(current_div)) << "[FATAL] Non-finite divergence encountered at step " << step;
        ASSERT_LT(current_div, 10.0) << "[FATAL] Divergence safety ceiling exceeded at step " << step 
                                    << " | Divergence Value: " << current_div;
    }

    // We compute total mass flow rates at the inlet and outlet faces:
    //      m_dot = sum(rho * u * dy * dz)
    double inlet_mass_flow = 0.0;
    double outlet_mass_flow = 0.0;
    double face_area = dy * dz;

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            size_t in_idx  = get_flat_index(1, j, k, nx, ny);
            size_t out_idx = get_flat_index(nx - 2, j, k, nx, ny);
            inlet_mass_flow  += density * u[in_idx]  * face_area;
            outlet_mass_flow += density * u[out_idx] * face_area;
        }
    }

    // We assert mass conservation between inlet and outlet:
    assert(std::abs(inlet_mass_flow - outlet_mass_flow) < 1e-2);
    ASSERT_NEAR(inlet_mass_flow, outlet_mass_flow, 1e-2)
        << "Mass flow conservation failure: Inlet mass rate does not match outlet mass rate.";
}
