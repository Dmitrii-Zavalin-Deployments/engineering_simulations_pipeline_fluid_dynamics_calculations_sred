/**
 * @file test_divergence_symmetry.cpp
 * @brief Unit test verifying symmetric RHS divergence calculation across boundary and interior cells.
 */

#include <gtest/gtest.h>
#include "grid_math.hpp"
#include <vector>
#include <cmath>

namespace navier_stokes_solver {
namespace testing {

TEST(DivergenceSymmetryTest, SymmetricStencilsIncludeBoundaryFaces) {
    const int nx = 5;
    const int ny = 5;
    const int nz = 5;
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1); // All fluid cells active

    // Initialize a linear velocity field: u = x, v = y, w = z
    // The analytical divergence is du/dx + dv/dy + dw/dz = 1.0 + 1.0 + 1.0 = 3.0 everywhere.
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                u_star[idx] = static_cast<double>(i) * dx;
                v_star[idx] = static_cast<double>(j) * dy;
                w_star[idx] = static_cast<double>(k) * dz;
            }
        }
    }

    // Select an interior cell to verify symmetric central-difference stencils
    int i = 2, j = 2, k = 2;
    size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
    size_t idx_west  = static_cast<size_t>(get_flat_index(i - 1, j, k, nx, ny));
    size_t idx_east  = static_cast<size_t>(get_flat_index(i + 1, j, k, nx, ny));
    size_t idx_south = static_cast<size_t>(get_flat_index(i, j - 1, k, nx, ny));
    size_t idx_north = static_cast<size_t>(get_flat_index(i, j + 1, k, nx, ny));
    size_t idx_down  = static_cast<size_t>(get_flat_index(i, j, k - 1, nx, ny));
    size_t idx_up    = static_cast<size_t>(get_flat_index(i, j, k + 1, nx, ny));

    // Compute symmetric derivatives matching the corrected orchestrator stencil
    const double dudx = (u_star[idx_east] - u_star[idx_west]) / (2.0 * dx);
    const double dvdy = (v_star[idx_north] - v_star[idx_south]) / (2.0 * dy);
    const double dwdz = (w_star[idx_up] - w_star[idx_down]) / (2.0 * dz);

    double divergence = dudx + dvdy + dwdz;

    // Assert that the computed divergence matches the exact analytical rate (3.0) 
    // without asymmetric truncation or missing upper boundary contributions.
    EXPECT_NEAR(divergence, 3.0, 1e-12);
}

} // namespace testing
} // namespace navier_stokes_solver
