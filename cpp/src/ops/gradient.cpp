#include "gradient.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

// Local 3D row-major indexing helper
inline size_t grad_idx(int i, int j, int k, int Ny, int Nz) {
    return static_cast<size_t>(i) * (Ny * Nz) + static_cast<size_t>(j) * Nz + k;
}

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

    #pragma omp parallel for collapse(3)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t c = grad_idx(i, j, k, Ny, Nz);

                // 1. Second-order central difference components: ∂p/∂x, ∂p/∂y, ∂p/∂z
                double gx = (field[grad_idx(i+1, j, k, Ny, Nz)] - field[grad_idx(i-1, j, k, Ny, Nz)]) / (2.0 * dx);
                double gy = (field[grad_idx(i, j+1, k, Ny, Nz)] - field[grad_idx(i, j-1, k, Ny, Nz)]) / (2.0 * dy);
                double gz = (field[grad_idx(i, j, k+1, Ny, Nz)] - field[grad_idx(i, j, k-1, Ny, Nz)]) / (2.0 * dz);

                // --- FORENSIC NUMERICAL AUDIT ---
                if (!std::isfinite(gx) || !std::isfinite(gy) || !std::isfinite(gz)) {
                    std::cerr << "MATH FAILURE: Gradient exploded at grid index [" 
                              << i << ", " << j << ", " << k << "] | "
                              << "Components: [" << gx << ", " << gy << ", " << gz << "]\n";
                    throw std::runtime_error("Pressure gradient is non-finite in grid computation.");
                }

                grad_x_out[c] = gx;
                grad_y_out[c] = gy;
                grad_z_out[c] = gz;
            }
        }
    }
}
