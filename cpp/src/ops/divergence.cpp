#include "divergence.hpp"

// Local 3D row-major indexing helper
inline size_t div_idx(int i, int j, int k, int Ny, int Nz) {
    return static_cast<size_t>(i) * (Ny * Nz) + static_cast<size_t>(j) * Nz + k;
}

void compute_divergence(
    const double* u_star, const double* v_star, const double* w_star,
    double* div_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz
) {
    if (dx == 0.0 || dy == 0.0 || dz == 0.0) {
        throw std::invalid_argument("GEOMETRY CRASH: Invalid zero dimensions provided for divergence calculation.");
    }

    #pragma omp parallel for collapse(3)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                size_t c = div_idx(i, j, k, Ny, Nz);

                // 1. Central difference components: ∂u*/∂x, ∂v*/∂y, ∂w*/∂z
                double div_x = (u_star[div_idx(i+1, j, k, Ny, Nz)] - u_star[div_idx(i-1, j, k, Ny, Nz)]) / (2.0 * dx);
                double div_y = (v_star[div_idx(i, j+1, k, Ny, Nz)] - v_star[div_idx(i, j-1, k, Ny, Nz)]) / (2.0 * dy);
                double div_z = (w_star[div_idx(i, j, k+1, Ny, Nz)] - w_star[div_idx(i, j, k-1, Ny, Nz)]) / (2.0 * dz);

                double divergence_val = div_x + div_y + div_z;

                // --- FORENSIC NUMERICAL AUDIT ---
                if (!std::isfinite(divergence_val)) {
                    std::cerr << "MATH FAILURE: Non-finite divergence at grid index [" 
                              << i << ", " << j << ", " << k << "] | "
                              << "Components [dx:" << div_x << ", dy:" << div_y << ", dz:" << div_z << "] | "
                              << "Result: " << divergence_val << "\n";
                    throw std::runtime_error("Divergence exploded. PPE source term is poisoned.");
                }

                div_out[c] = divergence_val;
            }
        }
    }
}
