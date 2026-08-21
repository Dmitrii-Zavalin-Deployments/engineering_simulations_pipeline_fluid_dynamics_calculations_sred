/**
 * @file test_corrector_momentum.cpp
 * @brief Literate unit test verifying that the pressure correction step preserves 
 *        internal fluid momentum near domain boundaries rather than zeroing it out.
 * 
 * WHAT: This test sets up a minimal 3x3x3 Cartesian grid where internal fluid 
 *       cells (k = 1) sit directly adjacent to non-fluid boundary walls (k = 0 and k = 2).
 * WHY:  An over-aggressive boundary check previously forced internal velocities to 
 *       zero whenever an adjacent neighbor had a mask != 1. This test ensures 
 *       that momentum propagates correctly and prevents frozen time evolution in 
 *       coarse or bounded simulation setups.
 */

#include <gtest/gtest.h>
#include "corrector.hpp"
#include "grid_math.hpp"
#include <vector>

namespace navier_stokes_solver {

TEST(CorrectorMomentumTest, PreservesInternalMomentumNearBoundaries) {
    // Define the dimensions of a coarse 3x3x3 grid to force all internal 
    // fluid nodes to sit directly adjacent to boundary planes.
    const int nx = 3;
    const int ny = 3;
    const int nz = 3;
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double rho = 1.0;

    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // Allocate solver velocity vectors.
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);

    // Populate trial velocities (w_star) with a uniform inflow field (1.0 m/s).
    // In a physically sound solver, this trial momentum must be preserved 
    // across the corrector step when no adverse pressure gradient opposes it.
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 1.0);

    // Initialize a uniform pressure field with zero gradients.
    std::vector<double> p(total_cells, 0.0);

    // Establish the computational cell mask:
    //   - mask == 1 represents active fluid cells.
    //   - mask == 0 represents solid boundary walls at the bottom (k = 0) and top (k = 2).
    std::vector<int> mask(total_cells, 1);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                if (k == 0 || k == nz - 1) {
                    mask[idx] = 0; // Solid boundary wall node
                }
            }
        }
    }

    // Execute the parallel corrector velocity projection step.
    EXPECT_NO_THROW({
        solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, dx, dy, dz, dt, rho
        );
    });

    // Verify the velocity outcome at the core internal fluid cell [i = 1, j = 1, k = 1].
    // The solved velocity component w should match the trial velocity w_star (1.0 m/s) 
    // rather than being incorrectly overridden to zero by boundary-neighbor conditions.
    size_t internal_idx = static_cast<size_t>(get_flat_index(1, 1, 1, nx, ny));

    // The expected velocity evaluation:
    //     w_final = w_star = 1.0
    EXPECT_NEAR(w[internal_idx], 1.0, 1e-5);
}

} // namespace navier_stokes_solver
