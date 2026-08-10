/**
 * @file laplacian.cpp
 * @brief Implementation of 3D Laplacian operator with OpenMP multi-threading.
 */

#include "laplacian.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#include "grid_math.hpp"
#endif

namespace navier_stokes_solver {

// Local 3D row-major indexing helper



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
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t c = lap_idx(i, j, k, Ny, Nz);

                double f_c  = field[c];
                double f_ip = field[lap_idx(i+1, j, k, Ny, Nz)];
                double f_im = field[lap_idx(i-1, j, k, Ny, Nz)];
                double f_jp = field[lap_idx(i, j+1, k, Ny, Nz)];
                double f_jm = field[lap_idx(i, j-1, k, Ny, Nz)];
                double f_kp = field[lap_idx(i, j, k+1, Ny, Nz)];
                double f_km = field[lap_idx(i, j, k-1, Ny, Nz)];

                double term_x = (f_ip - 2.0 * f_c + f_im) / dx2;
                double term_y = (f_jp - 2.0 * f_c + f_jm) / dy2;
                double term_z = (f_kp - 2.0 * f_c + f_km) / dz2;

                double lap_val = term_x + term_y + term_z;

                // --- FORENSIC NUMERICAL AUDIT ---
                if (!std::isfinite(lap_val)) {
                    #pragma omp critical
                    {
                        std::cerr << "MATH FAILURE: Non-finite Laplacian at grid index [" 
                                  << i << ", " << j << ", " << k << "] | "
                                  << "Center Val: " << f_c << " | "
                                  << "Terms [X:" << term_x << ", Y:" << term_y << ", Z:" << term_z << "]\n";
                        throw std::runtime_error("Laplacian exploded in grid computation.");
                    }
                }

                lap_out[c] = lap_val;
            }
        }
    }
}

} // namespace navier_stokes_solver