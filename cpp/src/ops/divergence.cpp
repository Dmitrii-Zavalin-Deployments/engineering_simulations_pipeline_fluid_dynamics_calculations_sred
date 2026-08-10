/**
 * @file divergence.cpp
 * @brief Implementation of 3D Divergence operator with OpenMP multi-threading.
 */

#include "divergence.hpp"
#include "grid_math.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void compute_divergence(
    const double* u_star, const double* v_star, const double* w_star,
    double* div_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
) {
    if (dx == 0.0 || dy == 0.0 || dz == 0.0) {
        throw std::invalid_argument("GEOMETRY CRASH: Invalid zero dimensions provided for divergence calculation.");
    }

    const long long total_cells = static_cast<long long>(Nx) * Ny * Nz;

    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t c = get_flat_index(i, j, k, Nx, Ny);

                // 1. Central difference components: ∂u*/∂x, ∂v*/∂y, ∂w*/∂z
                double div_x = (u_star[get_flat_index(i+1, j, k, Nx, Ny)] - u_star[get_flat_index(i-1, j, k, Nx, Ny)]) / (2.0 * dx);
                double div_y = (v_star[get_flat_index(i, j+1, k, Nx, Ny)] - v_star[get_flat_index(i, j-1, k, Nx, Ny)]) / (2.0 * dy);
                double div_z = (w_star[get_flat_index(i, j, k+1, Nx, Ny)] - w_star[get_flat_index(i, j, k-1, Nx, Ny)]) / (2.0 * dz);

                double divergence_val = div_x + div_y + div_z;

                // --- FORENSIC NUMERICAL AUDIT ---
                if (!std::isfinite(divergence_val)) {
                    #pragma omp critical
                    {
                        std::cerr << "MATH FAILURE: Non-finite divergence at grid index [" 
                                  << i << ", " << j << ", " << k << "] | "
                                  << "Components [dx:" << div_x << ", dy:" << div_y << ", dz:" << div_z << "] | "
                                  << "Result: " << divergence_val << "\n";
                        throw std::runtime_error("Divergence exploded. PPE source term is poisoned.");
                    }
                }

                div_out[c] = divergence_val;
            }
        }
    }
}

} // namespace navier_stokes_solver
