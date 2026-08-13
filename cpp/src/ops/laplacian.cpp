/**
 * @file laplacian.cpp
 * @brief Implementation of 3D Laplacian operator with OpenMP multi-threading and safe exception handling.
 */

#include "laplacian.hpp"
#include "grid_math.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void compute_laplacian(
    const double* field, double* lap_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
) {
    // Geometry Guard
    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        std::cerr << "GEOMETRY CRASH: Invalid grid dimensions provided for Laplacian calculation (dx=" 
                  << dx << ", dy=" << dy << ", dz=" << dz << ").\n";
        throw std::invalid_argument("Invalid grid geometry in Laplacian kernel.");
    }

    const long long total_cells = static_cast<long long>(Nx) * Ny * Nz;

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: laplacian.cpp | Operations (Cells): " << total_cells 
              << " | Grid: " << Nx << "x" << Ny << "x" << Nz 
              << " | Active Threads: " << active_threads << "\n";

    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    bool has_error = false;
    int err_i = 0, err_j = 0, err_k = 0;
    double err_val = 0.0;

    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int k = 1; k < Nz - 1; ++k) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                size_t c = get_flat_index(i, j, k, Nx, Ny);

                double f_c  = field[c];
                double f_ip = field[get_flat_index(i+1, j, k, Nx, Ny)];
                double f_im = field[get_flat_index(i-1, j, k, Nx, Ny)];
                double f_jp = field[get_flat_index(i, j+1, k, Nx, Ny)];
                double f_jm = field[get_flat_index(i, j-1, k, Nx, Ny)];
                double f_kp = field[get_flat_index(i, j, k+1, Nx, Ny)];
                double f_km = field[get_flat_index(i, j, k-1, Nx, Ny)];

                double term_x = (f_ip - 2.0 * f_c + f_im) / dx2;
                double term_y = (f_jp - 2.0 * f_c + f_jm) / dy2;
                double term_z = (f_kp - 2.0 * f_c + f_km) / dz2;

                double lap_val = term_x + term_y + term_z;

                // --- FORENSIC NUMERICAL AUDIT ---
                if (!std::isfinite(lap_val)) {
                    #pragma omp critical
                    {
                        if (!has_error) {
                            has_error = true;
                            err_i = i;
                            err_j = j;
                            err_k = k;
                            err_val = lap_val;
                        }
                    }
                }

                lap_out[c] = lap_val;
            }
        }
    }

    if (has_error) {
        std::cerr << "MATH FAILURE: Non-finite Laplacian detected at grid index [" 
                  << err_i << ", " << err_j << ", " << err_k << "] | Result: " << err_val << "\n";
        throw std::runtime_error("Laplacian exploded in grid computation.");
    }
}

} // namespace navier_stokes_solver
