#ifndef DIVERGENCE_HPP
#define DIVERGENCE_HPP

#include "base_operator.hpp"

namespace ops {

/**
 * Computes the scalar divergence field (∇ ⋅ v*) for the Pressure Poisson Equation (PPE)
 * using central differencing and strict finite-number audits.
 */
void compute_divergence(
    const double* u_star, const double* v_star, const double* w_star,
    double* div_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
);

} // namespace ops

#endif // DIVERGENCE_HPP
