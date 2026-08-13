/**
 * @file advection.cpp
 * @brief Implementation of 3D Advection operator with OpenMP multi-threading.
 */

#include "advection.hpp"
#include "grid_math.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void compute_advection(
    const double* u, const double* v, const double* w,
    const double* field, double* adv_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
) {
    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        throw std::invalid_argument("GEOMETRY CRASH: Invalid grid dimensions provided for advection calculation.");
    }

    const long long total_cells = static_cast<long long>(Nx) * Ny * Nz;

    bool has_error = false;
    int err_i = 0, err_j = 0, err_k = 0;
    double err_u = 0.0, err_v = 0.0, err_w = 0.0, err_val = 0.0;

    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t c = get_flat_index(i, j, k, Nx, Ny);

                double ui = u[c];
                double vi = v[c];
                double wi = w[c];

                // Central difference gradients for field
                double dfield_dx = (field[get_flat_index(i+1, j, k, Nx, Ny)] - field[get_flat_index(i-1, j, k, Nx, Ny)]) / (2.0 * dx);
                double dfield_dy = (field[get_flat_index(i, j+1, k, Nx, Ny)] - field[get_flat_index(i, j-1, k, Nx, Ny)]) / (2.0 * dy);
                double dfield_dz = (field[get_flat_index(i, j, k+1, Nx, Ny)] - field[get_flat_index(i, j, k-1, Nx, Ny)]) / (2.0 * dz);

                double advection_val = (ui * dfield_dx + vi * dfield_dy + wi * dfield_dz);

                // --- FORENSIC NUMERICAL AUDIT ---
                if (!std::isfinite(advection_val) || !std::isfinite(ui) || !std::isfinite(vi) || !std::isfinite(wi)) {
                    #pragma omp critical
                    {
                        if (!has_error) {
                            has_error = true;
                            err_i = i;
                            err_j = j;
                            err_k = k;
                            err_u = ui;
                            err_v = vi;
                            err_w = wi;
                            err_val = advection_val;
                        }
                    }
                }

                adv_out[c] = advection_val;
            }
        }
    }

    if (has_error) {
        std::cerr << "MATH FAILURE: Non-finite advection at grid index [" 
                  << err_i << ", " << err_j << ", " << err_k << "] | "
                  << "Vel: [" << err_u << ", " << err_v << ", " << err_w << "] | "
                  << "Result: " << err_val << "\n";
        throw std::runtime_error("Advection term exploded in grid computation.");
    }
}

} // namespace navier_stokes_solver
