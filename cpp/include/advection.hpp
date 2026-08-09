#ifndef ADVECTION_HPP
#define ADVECTION_HPP

#include "base_operator.hpp"

namespace navier_stokes_solver {

/**
 * Computes the 3D advection term: (v ⋅ ∇) * field
 * using central differencing and forensic finite-number auditing.
 */
void compute_advection(
    const double* u, const double* v, const double* w,
    const double* field, double* adv_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
);

} // namespace navier_stokes_solver

#endif // ADVECTION_HPP
