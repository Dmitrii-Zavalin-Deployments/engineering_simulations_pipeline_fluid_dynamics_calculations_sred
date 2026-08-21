/**
 * @file corrector.cpp
 * @brief Implementation of Step 4 Corrector Velocity Projection maintaining MAC face-centered
 *        gradients (dx) using boundary-conforming pressures from the Poisson/Neumann solver,
 *        with explicit no-penetration boundary condition enforcement at fluid-solid interfaces.
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
    // Keep 1-cell MAC face inverse grid spacing (dx, dy, dz) to prevent checkerboarding
    const double idx_inv = 1.0 / dx;
    const double idy_inv = 1.0 / dy;
    const double idz_inv = 1.0 / dz;

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

                // Compute adjacent neighbor indices for 1-cell MAC face pressure gradients and boundary checks
                const size_t idx_west  = static_cast<size_t>(get_flat_index(i - 1, j, k, nx, ny));
                const size_t idx_east  = static_cast<size_t>(get_flat_index(i + 1, j, k, nx, ny));
                const size_t idx_south = static_cast<size_t>(get_flat_index(i, j - 1, k, nx, ny));
                const size_t idx_north = static_cast<size_t>(get_flat_index(i, j + 1, k, nx, ny));
                const size_t idx_down  = static_cast<size_t>(get_flat_index(i, j, k - 1, nx, ny));
                const size_t idx_up    = static_cast<size_t>(get_flat_index(i, j, k + 1, nx, ny));

                const double p_center = p[idx_cell];
                
                // Use the actual neighbor pressures computed by apply_neumann_pressure.
                // Do NOT override with p_center, as wall pressures already contain the correct Neumann gradient.
                const double p_east  = p[idx_east];
                const double p_north = p[idx_north];
                const double p_up    = p[idx_up];

                // Compute 1-cell MAC face pressure gradients: dp/dx, dp/dy, dp/dz using dx
                const double dp_dx = (p_east - p_center) * idx_inv;
                const double dp_dy = (p_north - p_center) * idy_inv;
                const double dp_dz = (p_up - p_center) * idz_inv;

                // Project trial velocity onto divergence-free subspace (u^(n+1) = u* - (dt/rho) * grad(p))
                double new_u = u_star[idx_cell] - coeff * dp_dx;
                double new_v = v_star[idx_cell] - coeff * dp_dy;
                double new_w = w_star[idx_cell] - coeff * dp_dz;

                // Enforce explicit no-penetration boundary conditions at fluid-solid interfaces
                if (mask[idx_east] != 1 || mask[idx_west] != 1) new_u = 0.0;
                if (mask[idx_north] != 1 || mask[idx_south] != 1) new_v = 0.0;
                if (mask[idx_up] != 1 || mask[idx_down] != 1) new_w = 0.0;

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
