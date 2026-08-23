/**
 * @file divergence.cpp
 * @brief Implementation of 3D Divergence operator with OpenMP multi-threading and cache-optimized loop ordering.
 */

#include "divergence.hpp"
#include "grid_math.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>

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
    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        throw std::invalid_argument("GEOMETRY CRASH: Invalid grid dimensions provided for divergence calculation.");
    }

    const long long total_cells = static_cast<long long>(Nx) * Ny * Nz;

    // Zero-initialize output array to eliminate uninitialized boundary memory garbage
    std::fill_n(div_out, total_cells, 0.0);

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: divergence.cpp | Operations (Cells): " << total_cells 
              << " | Grid: " << Nx << "x" << Ny << "x" << Nz 
              << " | Active Threads: " << active_threads << "\n";

    bool has_error = false;
    int err_i = 0, err_j = 0, err_k = 0;
    double err_x = 0.0, err_y = 0.0, err_z = 0.0, err_val = 0.0;

    // Loop order optimized to k -> j -> i for row-major cache locality
    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int k = 1; k < Nz - 1; ++k) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
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
                        if (!has_error) {
                            has_error = true;
                            err_i = i;
                            err_j = j;
                            err_k = k;
                            err_x = div_x;
                            err_y = div_y;
                            err_z = div_z;
                            err_val = divergence_val;
                        }
                    }
                }

                div_out[c] = divergence_val;
            }
        }
    }

    if (has_error) {
        std::cerr << "MATH FAILURE: Non-finite divergence at grid index [" 
                  << err_i << ", " << err_j << ", " << err_k << "] | "
                  << "Components [dx:" << err_x << ", dy:" << err_y << ", dz:" << err_z << "] | "
                  << "Result: " << err_val << "\n";
        throw std::runtime_error("Divergence exploded. PPE source term is poisoned.");
    }
}

} // namespace navier_stokes_solver
