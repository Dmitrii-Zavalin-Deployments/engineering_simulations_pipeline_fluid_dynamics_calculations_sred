/**
 * @file corrector.cpp
 * @brief Implementation of Step 4 Corrector Velocity Projection for collocated grids with
 *        boundary-conforming one-sided pressure gradients at fluid-solid interfaces and
 *        stabilized 2nd-order central differences with anti-checkerboard damping in the interior.
 */

#include "corrector.hpp"
#include "grid_math.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void solve_corrector_parallel(
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    const std::vector<double>& u_star,
    const std::vector<double>& v_star,
    const std::vector<double>& w_star,
    const std::vector<double>& p,
    const std::vector<int>& mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho
) {
    if (nx < 3 || ny < 3 || nz < 3) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid dimensions must be at least 3x3x3 for corrector projection.");
    }
    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid spacing must be strictly positive.");
    }
    if (dt <= 0.0 || rho <= 0.0) {
        throw std::invalid_argument("PHYSICS ERROR: Time step dt and density rho must be strictly positive.");
    }

    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    if (u.size() != total_cells || v.size() != total_cells || w.size() != total_cells ||
        u_star.size() != total_cells || v_star.size() != total_cells || w_star.size() != total_cells ||
        p.size() != total_cells || mask.size() != total_cells) {
        throw std::invalid_argument("CONTRACT VIOLATION: Vector size mismatch in corrector module.");
    }

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: corrector.cpp | Operations (Cells): " << total_cells 
              << " | Grid: " << nx << "x" << ny << "x" << nz 
              << " | Active Threads: " << active_threads << "\n";

    const double coeff = dt / rho;
    const double idx_2inv = 0.5 / dx;
    const double idy_2inv = 0.5 / dy;
    const double idz_2inv = 0.5 / dz;

    bool has_error = false;
    int err_i = 0, err_j = 0, err_k = 0;
    double err_u = 0.0, err_v = 0.0, err_w = 0.0;

    // Execute corrector step strictly on active interior fluid cells (mask == 1)
    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                
                const size_t idx_cell = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                
                // Skip solid cells (mask == 0) and wall boundaries (mask == -1)
                if (mask[idx_cell] != 1) continue;

                // Neighbor indices
                const size_t idx_west  = static_cast<size_t>(get_flat_index(i - 1, j, k, nx, ny));
                const size_t idx_east  = static_cast<size_t>(get_flat_index(i + 1, j, k, nx, ny));
                const size_t idx_south = static_cast<size_t>(get_flat_index(i, j - 1, k, nx, ny));
                const size_t idx_north = static_cast<size_t>(get_flat_index(i, j + 1, k, nx, ny));
                const size_t idx_down  = static_cast<size_t>(get_flat_index(i, j, k - 1, nx, ny));
                const size_t idx_up    = static_cast<size_t>(get_flat_index(i, j, k + 1, nx, ny));

                const double p_center = p[idx_cell];
                const double p_west  = p[idx_west];
                const double p_east  = p[idx_east];
                const double p_south = p[idx_south];
                const double p_north = p[idx_north];
                const double p_down  = p[idx_down];
                const double p_up    = p[idx_up];

                // --- BOUNDARY-CONFORMING & STABILIZED PRESSURE GRADIENT EVALUATION ---
                
                // X-Direction Gradient
                double dp_dx;
                if (mask[idx_east] <= 0) {
                    dp_dx = (p_east - p_center) / dx; // One-sided backward toward east solid wall
                } else if (mask[idx_west] <= 0) {
                    dp_dx = (p_center - p_west) / dx; // One-sided forward toward west solid wall
                } else {
                    dp_dx = (p_east - p_west) * idx_2inv; // Interior 2nd-order central
                    const double lap_p_x = p_east - 2.0 * p_center + p_west;
                    dp_dx -= 0.5 * lap_p_x * idx_2inv;     // Anti-checkerboard stabilization
                }

                // Y-Direction Gradient
                double dp_dy;
                if (mask[idx_north] <= 0) {
                    dp_dy = (p_north - p_center) / dy; // One-sided backward toward north solid wall
                } else if (mask[idx_south] <= 0) {
                    dp_dy = (p_center - p_south) / dy; // One-sided forward toward south solid wall
                } else {
                    dp_dy = (p_north - p_south) * idy_2inv; // Interior 2nd-order central
                    const double lap_p_y = p_north - 2.0 * p_center + p_south;
                    dp_dy -= 0.5 * lap_p_y * idy_2inv;     // Anti-checkerboard stabilization
                }

                // Z-Direction Gradient
                double dp_dz;
                if (mask[idx_up] <= 0) {
                    dp_dz = (p_up - p_center) / dz; // One-sided backward toward upper solid wall
                } else if (mask[idx_down] <= 0) {
                    dp_dz = (p_center - p_down) / dz; // One-sided forward toward lower solid wall
                } else {
                    dp_dz = (p_up - p_down) * idz_2inv; // Interior 2nd-order central
                    const double lap_p_z = p_up - 2.0 * p_center + p_down;
                    dp_dz -= 0.5 * lap_p_z * idz_2inv;     // Anti-checkerboard stabilization
                }

                // Project trial velocity onto divergence-free subspace
                double new_u = u_star[idx_cell] - coeff * dp_dx;
                double new_v = v_star[idx_cell] - coeff * dp_dy;
                double new_w = w_star[idx_cell] - coeff * dp_dz;

                // --- FORENSIC NUMERICAL AUDIT ---
                if (!std::isfinite(new_u) || !std::isfinite(new_v) || !std::isfinite(new_w)) {
                    #pragma omp critical
                    {
                        if (!has_error) {
                            has_error = true;
                            err_i = i;
                            err_j = j;
                            err_k = k;
                            err_u = new_u;
                            err_v = new_v;
                            err_w = new_w;
                        }
                    }
                }

                u[idx_cell] = new_u;
                v[idx_cell] = new_v;
                w[idx_cell] = new_w;
            }
        }
    }

    if (has_error) {
        std::cerr << "MATH FAILURE: Non-finite velocity projected at grid index [" 
                  << err_i << ", " << err_j << ", " << err_k << "] | "
                  << "Vel: [" << err_u << ", " << err_v << ", " << err_w << "]\n";
        throw std::runtime_error("Corrector projection exploded. Velocity field is non-finite.");
    }
}

} // namespace navier_stokes_solver
