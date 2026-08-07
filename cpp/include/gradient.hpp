#ifndef GRADIENT_HPP
#define GRADIENT_HPP

#include "base_operator.hpp"

/**
 * Computes the 3D gradient vector of a scalar field (e.g., pressure p):
 * ∇p = (∂p/∂x, ∂p/∂y, ∂p/∂z) using central differencing and strict finite-number audits.
 */
void compute_gradient(
    const double* field,
    double* grad_x_out, double* grad_y_out, double* grad_z_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
);

#endif // GRADIENT_HPP
