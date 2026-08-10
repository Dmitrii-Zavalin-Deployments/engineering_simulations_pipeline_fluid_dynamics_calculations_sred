/**
 * @file gradient.cpp
 * @brief Implementation of 3D Gradient operator with OpenMP multi-threading.
 */

#include "gradient.hpp"
#include "grid_math.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void compute_gradient(
    const double* field,
    double* grad_x_out, double* grad_y_out, double* grad_z_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
) {
    // Geometry Guard (Rule 7 equivalent)
    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        std::cerr << "GEOMETRY CRASH: Invalid grid dimensions provided for gradient calculation (dx=" 
                  << dx << ", dy=" << dy << ", dz=" << dz << ").\n";
        throw std::invalid_argument("Invalid grid spacing in gradient kernel.");
    }

    const long long total_cells = static_cast<long long>(Nx) * Ny * Nz;

    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t c = get_flat_index(i, j, k, Nx, Ny);

                // 1. Second-order central difference components: ∂p/∂x, ∂p/∂y, ∂p/∂z
                double gx = (field[get_flat_index(i+1, j, k, Nx, Ny)] - field[get_flat_index(i-1, j, k, Nx, Ny)]) / (2.0 * dx);
                double gy = (field[get_flat_index(i, j+1, k, Nx, Ny)] - field[get_flat_index(i, j-1, k, Nx, Ny)]) / (2.0 * dy);
                double gz = (field[get_flat_index(i, j, k+1, Nx, Ny)] - field[get_flat_index(i, j, k-1, Nx, Ny)]) / (2.0 * dz);

                // --- FORENSIC NUMERICAL AUDIT ---
                if (!std::isfinite(gx) || !std::isfinite(gy) || !std::isfinite(gz)) {
                    #pragma omp critical
                    {
                        std::cerr << "MATH FAILURE: Gradient exploded at grid index [" 
                                  << i << ", " << j << ", " << k << "] | "
                                  << "Components: [" << gx << ", " << gy << ", " << gz << "]\n";
                        throw std::runtime_error("Pressure gradient is non-finite in grid computation.");
                    }
                }

                grad_x_out[c] = gx;
                grad_y_out[c] = gy;
                grad_z_out[c] = gz;
            }
        }
    }
}

} // namespace navier_stokes_solver
