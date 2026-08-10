/**
 * @file advection.cpp
 * @brief Implementation of 3D advection operator with OpenMP multi-threading.
 */

#include "advection.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#include "grid_math.hpp"
#endif

namespace navier_stokes_solver {

// Local 3D row-major indexing helper



void compute_advection(
    const double* u, const double* v, const double* w,
    const double* field, double* adv_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
) {
    const long long total_cells = static_cast<long long>(Nx) * Ny * Nz;

    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t c = advect_idx(i, j, k, Ny, Nz);

                // 1. Compute spatial derivatives (Central Differencing)
                double df_dx = (field[advect_idx(i+1, j, k, Ny, Nz)] - field[advect_idx(i-1, j, k, Ny, Nz)]) / (2.0 * dx);
                double df_dy = (field[advect_idx(i, j+1, k, Ny, Nz)] - field[advect_idx(i, j-1, k, Ny, Nz)]) / (2.0 * dy);
                double df_dz = (field[advect_idx(i, j, k+1, Ny, Nz)] - field[advect_idx(i, j, k-1, Ny, Nz)]) / (2.0 * dz);

                // 2. Fetch cell-centered velocities
                double u_c = u[c];
                double v_c = v[c];
                double w_c = w[c];

                // 3. Assemble advection term: (v ⋅ ∇)f
                double advection_val = (u_c * df_dx) + (v_c * df_dy) + (w_c * df_dz);

                // --- FORENSIC NUMERICAL AUDIT ---
                if (!std::isfinite(advection_val)) {
                    #pragma omp critical
                    {
                        std::cerr << "MATH FAILURE: Non-finite advection at grid index [" 
                                  << i << ", " << j << ", " << k << "] | "
                                  << "Vel: [" << u_c << ", " << v_c << ", " << w_c << "] | "
                                  << "Gradients: [" << df_dx << ", " << df_dy << ", " << df_dz << "]\n";
                        throw std::runtime_error("Advection term exploded in grid computation.");
                    }
                }

                adv_out[c] = advection_val;
            }
        }
    }
}

} // namespace navier_stokes_solver