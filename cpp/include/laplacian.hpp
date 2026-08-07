#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include "base_operator.hpp"

namespace ops {

/**
 * Computes the 3D discrete Laplacian field: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
 * using a 7-point stencil and strict finite-number audits.
 */
void compute_laplacian(
    const double* field, double* lap_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
);

} // namespace ops

#endif // LAPLACIAN_HPP
