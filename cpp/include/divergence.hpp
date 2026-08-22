/**
 * @file divergence.hpp
 * @brief Header for scalar velocity divergence calculation used in the
 *        Pressure Poisson Equation (PPE).
 *
 *        Computes:
 *            ∇·v* = ∂u*/∂x + ∂v*/∂y + ∂w*/∂z
 *
 *        Implementation (in .cpp) guarantees:
 *          - Interior central differencing
 *          - Boundary‑safe stencil handling
 *          - Finite‑number auditing
 */

#ifndef DIVERGENCE_HPP
#define DIVERGENCE_HPP

#include "base_operator.hpp"
#include "grid_math.hpp"

namespace navier_stokes_solver {

/**
 * @brief Computes the scalar divergence field (∇·v*) for the PPE.
 *
 * @param u_star   Predictor x‑velocity
 * @param v_star   Predictor y‑velocity
 * @param w_star   Predictor z‑velocity
 * @param div_out  Output divergence field
 * @param Nx,Ny,Nz Grid resolution
 * @param dx,dy,dz Grid spacing
 *
 * Notes:
 *   - Operates on interior cells (1 … Nx‑2, etc.)
 *   - Boundary safety is handled inside the .cpp implementation
 *   - No mask is required here; upstream modules guarantee valid boundary states
 */
void compute_divergence(
    const double* u_star,
    const double* v_star,
    const double* w_star,
    double* div_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
);

} // namespace navier_stokes_solver

#endif // DIVERGENCE_HPP

