/**
 * @file advection.hpp
 * @brief Header for 3D Advection operator computation.
 *
 *        Computes the nonlinear convective term:
 *
 *              (u, v, w) ⋅ ∇field
 *
 *        using:
 *          - interior central differencing
 *          - boundary‑safe one‑sided stencils (handled in .cpp)
 *          - finite‑number auditing
 *
 *        This operator is intentionally mask‑agnostic at the header level.
 *        Boundary and solid safety is enforced inside the implementation.
 */

#ifndef ADVECTION_HPP
#define ADVECTION_HPP

#include "base_operator.hpp"

namespace navier_stokes_solver {

/**
 * @brief Computes the 3D advection term (v ⋅ ∇)field.
 *
 * @param u      Cell‑centered x‑velocity
 * @param v      Cell‑centered y‑velocity
 * @param w      Cell‑centered z‑velocity
 * @param field  Scalar field to be advected
 * @param adv_out Output advection term
 * @param Nx,Ny,Nz Grid resolution
 * @param dx,dy,dz Grid spacing
 *
 * Notes:
 *   - Operates on interior cells (1 … Nx‑2, etc.)
 *   - Boundary safety is handled inside the .cpp implementation
 *   - No mask is required here; upstream modules guarantee valid boundary states
 */
void compute_advection(
    const double* u,
    const double* v,
    const double* w,
    const double* field,
    double* adv_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
);

} // namespace navier_stokes_solver

#endif // ADVECTION_HPP

