/**
 * @file corrector.cpp
 * @brief Implementation of Step 4 Corrector Velocity Projection for staggered MAC grid.
 */

#include "corrector.hpp"
#include "grid_math.hpp"
#include <stdexcept>

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
    const size_t u_size = static_cast<size_t>(nx + 1) * ny * nz;
    const size_t v_size = static_cast<size_t>(nx) * (ny + 1) * nz;
    const size_t w_size = static_cast<size_t>(nx) * ny * (nz + 1);

    if (u.size() != u_size || v.size() != v_size || w.size() != w_size ||
        u_star.size() != u_size || v_star.size() != v_size || w_star.size() != w_size ||
        p.size() != total_cells || mask.size() != total_cells) {
        throw std::invalid_argument("CONTRACT VIOLATION: Vector size mismatch for staggered MAC grid in corrector module.");
    }

    const double coeff_dx = dt / (rho * dx);
    const double coeff_dy = dt / (rho * dy);
    const double coeff_dz = dt / (rho * dz);

    // 1. Update u-velocities at x-faces (i from 1 to nx - 1)
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                size_t u_idx = static_cast<size_t>(i) + j * (nx + 1) + k * (nx + 1) * ny;
                size_t p_left  = static_cast<size_t>(i - 1) + j * nx + k * nx * ny;
                size_t p_right = static_cast<size_t>(i)     + j * nx + k * nx * ny;

                if (mask[p_left] == 1 && mask[p_right] == 1) {
                    double dp_dx = p[p_right] - p[p_left];
                    u[u_idx] = u_star[u_idx] - coeff_dx * dp_dx;
                } else {
                    u[u_idx] = u_star[u_idx];
                }
            }
        }
    }

    // 2. Update v-velocities at y-faces (j from 1 to ny - 1)
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 1; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t v_idx = static_cast<size_t>(i) + j * nx + k * nx * (ny + 1);
                size_t p_south = static_cast<size_t>(i) + (j - 1) * nx + k * nx * ny;
                size_t p_north = static_cast<size_t>(i) + j * nx + k * nx * ny;

                if (mask[p_south] == 1 && mask[p_north] == 1) {
                    double dp_dy = p[p_north] - p[p_south];
                    v[v_idx] = v_star[v_idx] - coeff_dy * dp_dy;
                } else {
                    v[v_idx] = v_star[v_idx];
                }
            }
        }
    }

    // 3. Update w-velocities at z-faces (k from 1 to nz - 1)
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 1; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t w_idx = static_cast<size_t>(i) + j * nx + k * nx * ny;
                size_t p_down = static_cast<size_t>(i) + j * nx + (k - 1) * nx * ny;
                size_t p_up   = static_cast<size_t>(i) + j * nx + k * nx * ny;

                if (mask[p_down] == 1 && mask[p_up] == 1) {
                    double dp_dz = p[p_up] - p[p_down];
                    w[w_idx] = w_star[w_idx] - coeff_dz * dp_dz;
                } else {
                    w[w_idx] = w_star[w_idx];
                }
            }
        }
    }
}

} // namespace navier_stokes_solver
