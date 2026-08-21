/**
 * @file test_solid_neumann_pressure.cpp
 * @brief Unit test verifying zero-normal pressure gradient enforcement on solid cells (mask == 0).
 */

#include <gtest/gtest.h>
#include "pressure_poisson_solver.hpp"
#include "grid_math.hpp"
#include <vector>

namespace navier_stokes_solver {
namespace testing {

TEST(PressurePoissonTest, SolidNeumannBoundaryZeroNormalGradient) {
    // 1. Setup a small 3x3x3 grid
    const int nx = 3;
    const int ny = 3;
    const int nz = 3;
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 0); // Default all cells to solid/wall

    // 2. Set center cell (1, 1, 1) as active fluid (mask == 1) with a test pressure
    const size_t center_idx = static_cast<size_t>(get_flat_index(1, 1, 1, nx, ny));
    mask[center_idx] = 1;
    p[center_idx] = 42.0; 

    // 3. Set adjacent east cell (2, 1, 1) as solid (mask == 0) with a dummy pressure
    const size_t east_idx = static_cast<size_t>(get_flat_index(2, 1, 1, nx, ny));
    mask[east_idx] = 0;
    p[east_idx] = 0.0; 

    // 4. Apply the updated solid Neumann pressure routine
    apply_solid_neumann_pressure_parallel(p, mask, nx, ny, nz, dx, dy, dz);

    // 5. Assert that the solid cell pressure matches the adjacent fluid cell 
    // to enforce zero-normal pressure gradient (dp/dn = 0) instead of harmonic decay.
    EXPECT_NEAR(p[east_idx], p[center_idx], 1e-12);
}

} // namespace testing
} // namespace navier_stokes_solver
