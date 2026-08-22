/**
 * @file laplacian.hpp
 * @brief Header for 3D discrete Laplacian operator using a 7‑point stencil.
 *
 *        Computes:
 *            ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
 *
 *        Implementation (in .cpp) guarantees:
 *          - Interior central differencing
 *          - Boundary‑safe stencil handling
 *          - Finite‑number auditing
 */

#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include "base_operator.hpp"
#include "grid_math.hpp"

namespace navier_stokes_solver {

/**
 * @brief Computes the 3D discrete Laplacian of a scalar field using
 *        a 7‑point finite‑difference stencil.
 *
 * @param field    Input scalar field
 * @param lap_out  Output Laplacian field
 * @param Nx,Ny,Nz Grid resolution
 * @param dx,dy,dz Grid spacing
 *
 * Notes:
 *   - Operates on interior cells (1 … Nx‑2, etc.)
 *   - Boundary safety is handled inside the .cpp implementation
 *   - No mask is required here; upstream modules guarantee valid boundary states
 */
void compute_laplacian(
    const double* field,
    double* lap_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
);

} // namespace navier_stokes_solver

#endif // LAPLACIAN_HPP

