/**
 * @file corrector.cpp
 * @brief Implementation of Step 4 Corrector Velocity Projection.
 */

#include "corrector.hpp"
#include "grid_math.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

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
    double dt, double rho)
{
    const double coeff = dt / rho;
    const double idx_inv = 1.0 / (2.0 * dx);
    const double idy_inv = 1.0 / (2.0 * dy);
    const double idz_inv = 1.0 / (2.0 * dz);

    // Execute corrector step strictly on active interior fluid cells (mask == 1)
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                
                const size_t idx_cell = static_cast<size_t>(navier_stokes_solver::get_flat_index(i, j, k, nx, ny));
                
                // Skip solid cells (mask == 0) and wall boundaries (mask == -1)
                if (mask[idx_cell] != 1) continue;

                // Compute neighbor indices via unified grid_math SSoT
                const size_t idx_west  = static_cast<size_t>(navier_stokes_solver::get_flat_index(i - 1, j, k, nx, ny));
                const size_t idx_east  = static_cast<size_t>(navier_stokes_solver::get_flat_index(i + 1, j, k, nx, ny));
                const size_t idx_south = static_cast<size_t>(navier_stokes_solver::get_flat_index(i, j - 1, k, nx, ny));
                const size_t idx_north = static_cast<size_t>(navier_stokes_solver::get_flat_index(i, j + 1, k, nx, ny));
                const size_t idx_down  = static_cast<size_t>(navier_stokes_solver::get_flat_index(i, j, k - 1, nx, ny));
                const size_t idx_up    = static_cast<size_t>(navier_stokes_solver::get_flat_index(i, j, k + 1, nx, ny));

                // Compute central pressure gradients: dp/dx, dp/dy, dp/dz
                const double p_west  = p[idx_west];
                const double p_east  = p[idx_east];
                const double p_south = p[idx_south];
                const double p_north = p[idx_north];
                const double p_down  = p[idx_down];
                const double p_up    = p[idx_up];

                const double dp_dx = (p_east - p_west) * idx_inv;
                const double dp_dy = (p_north - p_south) * idy_inv;
                const double dp_dz = (p_up - p_down) * idz_inv;

                // Project trial velocity onto divergence-free subspace (u^(n+1) = u* - (dt/rho) * grad(p))
                u[idx_cell] = u_star[idx_cell] - coeff * dp_dx;
                v[idx_cell] = v_star[idx_cell] - coeff * dp_dy;
                w[idx_cell] = w_star[idx_cell] - coeff * dp_dz;
            }
        }
    }
}
