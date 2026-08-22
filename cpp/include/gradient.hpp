/**
 * @file gradient.hpp
 * @brief Header for 3D gradient calculation of scalar fields (e.g., pressure).
 *
 *        Computes:
 *            ∇field = (∂field/∂x, ∂field/∂y, ∂field/∂z)
 *
 *        Implementation (in .cpp) guarantees:
 *          - Interior central differencing
 *          - Boundary‑safe one‑sided stencils
 *          - Finite‑number auditing
 */

#ifndef GRADIENT_HPP
#define GRADIENT_HPP

#include "base_operator.hpp"
#include "grid_math.hpp"

namespace navier_stokes_solver {

/**
 * @brief Computes the 3D gradient vector of a scalar field.
 *
 * @param field        Input scalar field
 * @param grad_x_out   Output ∂field/∂x
 * @param grad_y_out   Output ∂field/∂y
 * @param grad_z_out   Output ∂field/∂z
 * @param Nx,Ny,Nz     Grid resolution
 * @param dx,dy,dz     Grid spacing
 *
 * Notes:
 *   - Operates on interior cells (1 … Nx‑2, etc.)
 *   - Boundary safety handled inside the .cpp implementation
 *   - No mask required; upstream modules guarantee valid boundary states
 */
void compute_gradient(
    const double* field,
    double* grad_x_out,
    double* grad_y_out,
    double* grad_z_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
);

} // namespace navier_stokes_solver

#endif // GRADIENT_HPP