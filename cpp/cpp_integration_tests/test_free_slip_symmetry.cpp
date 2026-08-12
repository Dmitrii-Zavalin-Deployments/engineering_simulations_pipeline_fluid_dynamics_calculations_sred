/**
 * @file test_free_slip_symmetry.cpp
 * @brief Scenario 3.4: Free-Slip Symmetry Plane Boundary Verification
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "orchestrator.hpp"
#include "predictor.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

TEST(BoundaryConditionsTest, FreeSlipSymmetryPlane) {
    int nx = 6, ny = 8, nz = 6;
    double dx = 0.1, dy = 0.1, dz = 0.1;
    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    double density = 1000.0;
    double mu = 0.001;
    double dt = 0.001;

    std::vector<int> mask(total_cells, 1);
    std::vector<double> u(total_cells, 1.5);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    std::vector<BoundaryCondition> bc_list;
    BoundaryCondition bc_sym;
    bc_sym.location = "y_max";
    bc_sym.type = "free_slip_symmetry";
    bc_sym.v_val = 0.0;
    bc_sym.values.has_u = false;
    bc_sym.values.has_v = true; bc_sym.values.v = 0.0;
    bc_sym.values.has_w = false;
    bc_sym.values.has_p = false;
    bc_list.push_back(bc_sym);

    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

    int top_j = ny - 1;
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            size_t boundary_idx = get_flat_index(i, top_j,     k, nx, ny);
            size_t interior_idx = get_flat_index(i, top_j - 1, k, nx, ny);
            EXPECT_DOUBLE_EQ(u[boundary_idx], u[interior_idx])
                << "Free-slip symmetry failure: Tangential velocity gradient du/dy is non-zero across y_max boundary.";
        }
    }
}
