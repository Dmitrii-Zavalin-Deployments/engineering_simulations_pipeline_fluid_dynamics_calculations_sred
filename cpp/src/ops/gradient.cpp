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
    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        throw std::invalid_argument("GEOMETRY CRASH: Invalid grid dimensions provided for gradient calculation.");
    }

    const long long total_cells = static_cast<long long>(Nx) * Ny * Nz;

    bool has_error = false;
    int err_i = 0, err_j = 0, err_k = 0;
    double err_gx = 0.0, err_gy = 0.0, err_gz = 0.0;

    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t c = get_flat_index(i, j, k, Nx, Ny);

                // Central difference components: ∂field/∂x, ∂field/∂y, ∂field/∂z
                double gx = (field[get_flat_index(i+1, j, k, Nx, Ny)] - field[get_flat_index(i-1, j, k, Nx, Ny)]) / (2.0 * dx);
                double gy = (field[get_flat_index(i, j+1, k, Nx, Ny)] - field[get_flat_index(i, j-1, k, Nx, Ny)]) / (2.0 * dy);
                double gz = (field[get_flat_index(i, j, k+1, Nx, Ny)] - field[get_flat_index(i, j, k-1, Nx, Ny)]) / (2.0 * dz);

                // --- FORENSIC NUMERICAL AUDIT ---
                if (!std::isfinite(gx) || !std::isfinite(gy) || !std::isfinite(gz)) {
                    #pragma omp critical
                    {
                        if (!has_error) {
                            has_error = true;
                            err_i = i;
                            err_j = j;
                            err_k = k;
                            err_gx = gx;
                            err_gy = gy;
                            err_gz = gz;
                        }
                    }
                }

                grad_x_out[c] = gx;
                grad_y_out[c] = gy;
                grad_z_out[c] = gz;
            }
        }
    }

    if (has_error) {
        std::cerr << "MATH FAILURE: Gradient exploded at grid index [" 
                  << err_i << ", " << err_j << ", " << err_k << "] | "
                  << "Components: [" << err_gx << ", " << err_gy << ", " << err_gz << "]\n";
        throw std::runtime_error("Pressure gradient is non-finite in grid computation.");
    }
}

} // namespace navier_stokes_solver
