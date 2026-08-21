/**
 * @file test_solid_neumann_pressure.cpp
 * @brief Unit test verifying zero-normal pressure gradient enforcement on solid cells (mask == 0).
 * 
 * ============================================================================
 * WHAT THIS TEST IS EVALUATING:
 * ============================================================================
 * This unit test validates that `apply_solid_neumann_pressure_parallel()` correctly 
 * enforces the zero-normal pressure gradient condition ($\frac{\partial P}{\partial n} = 0$) 
 * on solid cells ($mask == 0$). Specifically, it checks that a solid cell directly copies 
 * the pressure value of the adjacent active fluid cell rather than solving Laplace's equation 
 * via 6-point harmonic averaging.
 * 
 * ============================================================================
 * WHY THIS TEST IS CRITICAL:
 * ============================================================================
 * Solving Laplace's equation inside solid cells causes uninitialized or zero boundary 
 * values to bleed into the solid domain ($P_{\text{solid}} \approx \frac{1}{6} P_{\text{fluid}}$), 
 * corrupting the pressure field and generating severe numerical instabilities and pressure blow-ups. 
 * This test guarantees Neumann boundary integrity.
 */

#include <gtest/gtest.h>
#include "pressure_poisson_solver.hpp"
#include "grid_math.hpp"
#include <vector>
#include <cmath>

namespace navier_stokes_solver {

TEST(PressurePoissonTest, SolidNeumannBoundaryZeroNormalGradient) {
    // We define a small 3x3x3 Cartesian grid domain to test local solid Neumann handling.
    const int nx = 3;
    const int ny = 3;
    const int nz = 3;
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 0); // Default all cells to solid/wall (mask == 0).

    // We designate the center cell (1, 1, 1) as an active fluid cell (mask == 1) 
    // and assign a baseline test pressure:
    //     p_fluid = 42.0 Pa
    const size_t center_idx = static_cast<size_t>(get_flat_index(1, 1, 1, nx, ny));
    mask[center_idx] = 1;
    p[center_idx] = 42.0; 

    // We designate the adjacent east cell (2, 1, 1) as a solid cell (mask == 0) 
    // initialized with a dummy pressure value of 0.0 Pa.
    const size_t east_idx = static_cast<size_t>(get_flat_index(2, 1, 1, nx, ny));
    mask[east_idx] = 0;
    p[east_idx] = 0.0; 

    // We apply the solid Neumann boundary condition routine to propagate pressures:
    //     P_solid = P_adjacent_fluid
    apply_solid_neumann_pressure_parallel(p, mask, nx, ny, nz, dx, dy, dz);

    // Under the zero-normal gradient condition ($\frac{\partial P}{\partial n} = 0$), the solid cell pressure 
    // must exactly match the adjacent fluid cell pressure without harmonic decay:
    //     assert abs(p_solid - p_fluid) < 1e-12
    ASSERT_NEAR(p[east_idx], p[center_idx], 1e-12);
}

} // namespace navier_stokes_solver