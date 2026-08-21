/**
 * @file test_free_slip_symmetry.cpp
 * @brief Scenario 3.4: Free-Slip Symmetry Plane Boundary Verification
 *
 * LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
 * ---------------------------------------------------------------------------------
 * Free-slip symmetry boundaries enforce zero normal velocity and zero shear stress
 * gradients across the boundary plane. For a boundary aligned with the y-axis (y_max):
 * 
 *     v_normal = v = 0.0
 *     du/dy = 0, dw/dy = 0
 * 
 * TEST SCENARIO:
 *   - We initialize the domain with a non-zero normal velocity field (v = 1.0 m/s) 
 *     to verify that the pre-step boundary enforcement correctly projects or zeroes out 
 *     the normal velocity component at the y_max symmetry plane.
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cassert>
#include "orchestrator.hpp"
#include "predictor.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

TEST(BoundaryConditionsTest, FreeSlipSymmetryPlane) {
    // We define the grid dimensions and spatial resolution:
    //     nx = 6, ny = 8, nz = 6
    //     dx = 0.1 m, dy = 0.1 m, dz = 0.1 m
    int nx = 6, ny = 8, nz = 6;
    double dx = 0.1, dy = 0.1, dz = 0.1;
    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // We define physical fluid properties and time-step parameters:
    //     density = 1000.0 kg/m^3, mu = 0.001 Pa*s, dt = 0.001 s
    double density = 1000.0;
    double mu = 0.001;
    double dt = 0.001;

    // We initialize the fluid mask and velocity fields:
    //     mask = 1 (active fluid domain)
    //     u = 1.5 m/s, v = 1.0 m/s (non-zero initial normal velocity), w = 0.0 m/s, p = 0.0 Pa
    std::vector<int> mask(total_cells, 1);
    std::vector<double> u(total_cells, 1.5);
    std::vector<double> v(total_cells, 1.0); 
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // We configure the free-slip symmetry boundary condition at the y_max boundary:
    //     v_val = 0.0
    std::vector<BoundaryCondition> bc_list;
    BoundaryCondition bc_sym;
    bc_sym.location = "y_max";
    bc_sym.type = "free-slip"; // Corrected to match the JSON schema enum standard
    bc_sym.v_val = 0.0;
    bc_sym.values.has_u = false;
    bc_sym.values.has_v = true; bc_sym.values.v = 0.0;
    bc_sym.values.has_w = false;
    bc_sym.values.has_p = false;
    bc_list.push_back(bc_sym);

    // We execute the simulation pre-step to enforce boundary conditions:
    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

    // We verify that the normal velocity v is exactly zero across all cells at the y_max boundary:
    int top_j = ny - 1;
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            size_t boundary_idx = get_flat_index(i, top_j, k, nx, ny);
            ASSERT_DOUBLE_EQ(v[boundary_idx], 0.0)
                << "Free-slip symmetry failure: Normal velocity v is non-zero at y_max symmetry boundary.";
        }
    }
}
