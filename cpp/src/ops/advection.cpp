/**
 * @file advection.cpp
 * @brief Mask-aware 3D Advection operator with boundary-safe stencils and cache-optimized loop ordering.
 */

#include "advection.hpp"
#include "grid_math.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>

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
        throw std::invalid_argument("GEOMETRY ERROR: dx, dy, dz must be strictly positive.");
    }

    const long long total_cells = static_cast<long long>(Nx) * Ny * Nz;

    // Zero-initialize output array to eliminate uninitialized boundary memory garbage
    std::fill_n(adv_out, total_cells, 0.0);

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: advection.cpp | Operations (Cells): "
              << total_cells << " | Grid: "
              << Nx << "x" << Ny << "x" << Nz
              << " | Active Threads: " << active_threads << "\n";

    bool has_error = false;
    int err_i = 0, err_j = 0, err_k = 0;
    double err_val = 0.0;

    // -------------------------------------------------------------------------
    // IMPORTANT:
    // Advection must NOT read boundary garbage. We compute only on interior cells.
    // Loop order optimized to k -> j -> i for row-major cache locality.
    // -------------------------------------------------------------------------

    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int k = 1; k < Nz - 1; ++k) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {

                size_t c = get_flat_index(i, j, k, Nx, Ny);

                // Velocity at cell center
                double ui = u[c];
                double vi = v[c];
                double wi = w[c];

                // -----------------------------
                // Boundary-safe central differencing
                // -----------------------------

                size_t idx_e = get_flat_index(i + 1, j, k, Nx, Ny);
                size_t idx_w = get_flat_index(i - 1, j, k, Nx, Ny);
                size_t idx_n = get_flat_index(i, j + 1, k, Nx, Ny);
                size_t idx_s = get_flat_index(i, j - 1, k, Nx, Ny);
                size_t idx_t = get_flat_index(i, j, k + 1, Nx, Ny);
                size_t idx_b = get_flat_index(i, j, k - 1, Nx, Ny);

                double dfield_dx = (field[idx_e] - field[idx_w]) / (2.0 * dx);
                double dfield_dy = (field[idx_n] - field[idx_s]) / (2.0 * dy);
                double dfield_dz = (field[idx_t] - field[idx_b]) / (2.0 * dz);

                double adv_val = ui * dfield_dx + vi * dfield_dy + wi * dfield_dz;

                // -----------------------------
                // Numerical safety audit
                // -----------------------------
                if (!std::isfinite(adv_val)) {
                    #pragma omp critical
                    {
                        if (!has_error) {
                            has_error = true;
                            err_i = i;
                            err_j = j;
                            err_k = k;
                            err_val = adv_val;
                        }
                    }
                }

                adv_out[c] = adv_val;
            }
        }
    }

    if (has_error) {
        std::cerr << "MATH FAILURE: Non-finite advection at grid index ["
                  << err_i << ", " << err_j << ", " << err_k
                  << "] | Value: " << err_val << "\n";
        throw std::runtime_error("Advection operator produced non-finite values.");
    }
}

} // namespace navier_stokes_solver
