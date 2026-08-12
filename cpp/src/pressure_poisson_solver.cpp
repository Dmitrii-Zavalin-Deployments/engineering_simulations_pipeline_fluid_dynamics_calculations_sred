/**
 * @file pressure_poisson_solver.cpp
 * @brief Implementation of Step 3 Pressure Poisson Solver (Red-Black GS) with robust safety validation
 *        and hydrostatic pressure / body-force boundary balancing.
 */

#include "pressure_poisson_solver.hpp"
#include "grid_math.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void apply_neumann_pressure(
    std::vector<double>& p,
    const std::string& location,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double density,
    const std::vector<double>& gravity
) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid spacing must be strictly positive in Neumann application.");
    }

    if (gravity.size() != 3) {
        throw std::invalid_argument("CONTRACT VIOLATION: gravity vector must contain exactly 3 components [gx, gy, gz].");
    }

    const double gx = gravity[0];
    const double gy = gravity[1];
    const double gz = gravity[2];

    double dp_dx = density * gx;
    double dp_dy = density * gy;
    double dp_dz = density * gz;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));

                if (location == "x_min" && i == 0) {
                    p[idx] = p[static_cast<size_t>(get_flat_index(1, j, k, nx, ny))] - dp_dx * dx;
                } else if (location == "x_max" && i == nx - 1) {
                    p[idx] = p[static_cast<size_t>(get_flat_index(nx - 2, j, k, nx, ny))] + dp_dx * dx;
                } else if (location == "y_min" && j == 0) {
                    p[idx] = p[static_cast<size_t>(get_flat_index(i, 1, k, nx, ny))] - dp_dy * dy;
                } else if (location == "y_max" && j == ny - 1) {
                    p[idx] = p[static_cast<size_t>(get_flat_index(i, ny - 2, k, nx, ny))] + dp_dy * dy;
                } else if (location == "z_min" && k == 0) {
                    p[idx] = p[static_cast<size_t>(get_flat_index(i, j, 1, nx, ny))] - dp_dz * dz;
                } else if (location == "z_max" && k == nz - 1) {
                    p[idx] = p[static_cast<size_t>(get_flat_index(i, j, nz - 2, nx, ny))] + dp_dz * dz;
                } else if (location == "wall") {
                    if (i == 0)         p[idx] = p[static_cast<size_t>(get_flat_index(1, j, k, nx, ny))] - dp_dx * dx;
                    else if (i == nx - 1) p[idx] = p[static_cast<size_t>(get_flat_index(nx - 2, j, k, nx, ny))] + dp_dx * dx;
                    else if (j == 0)      p[idx] = p[static_cast<size_t>(get_flat_index(i, 1, k, nx, ny))] - dp_dy * dy;
                    else if (j == ny - 1) p[idx] = p[static_cast<size_t>(get_flat_index(i, ny - 2, k, nx, ny))] + dp_dy * dy;
                    else if (k == 0)      p[idx] = p[static_cast<size_t>(get_flat_index(i, j, 1, nx, ny))] - dp_dz * dz;
                    else if (k == nz - 1) p[idx] = p[static_cast<size_t>(get_flat_index(i, j, nz - 2, nx, ny))] + dp_dz * dz;
                }
            }
        }
    }
}

void apply_solid_neumann_pressure_parallel(
    std::vector<double>& p, 
    const std::vector<int>& mask, 
    int nx, int ny, int nz
) {
    if (nx <= 2 || ny <= 2 || nz <= 2) return;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                if (mask[idx] != 0) continue; // Target internal solid cells only

                const size_t idx_west  = static_cast<size_t>(get_flat_index(i - 1, j, k, nx, ny));
                const size_t idx_east  = static_cast<size_t>(get_flat_index(i + 1, j, k, nx, ny));
                const size_t idx_south = static_cast<size_t>(get_flat_index(i, j - 1, k, nx, ny));
                const size_t idx_north = static_cast<size_t>(get_flat_index(i, j + 1, k, nx, ny));
                const size_t idx_down  = static_cast<size_t>(get_flat_index(i, j, k - 1, nx, ny));
                const size_t idx_up    = static_cast<size_t>(get_flat_index(i, j, k + 1, nx, ny));

                p[idx] = (p[idx_east] + p[idx_west] + p[idx_north] + p[idx_south] + p[idx_up] + p[idx_down]) / 6.0;
            }
        }
    }
}

void solve_poisson_red_black_parallel(
    std::vector<double>& p,
    const std::vector<double>& rhs,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    int max_iters, double tol,
    double density,
    const std::vector<double>& gravity
) {
    if (nx < 3 || ny < 3 || nz < 3) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid dimensions must be at least 3x3x3 for Poisson solver.");
    }
    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid spacing must be strictly positive.");
    }
    if (max_iters <= 0 || tol < 0.0) {
        throw std::invalid_argument("ITERATION ERROR: Invalid max iterations or tolerance.");
    }

    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    if (p.size() != total_cells || rhs.size() != total_cells || mask.size() != total_cells) {
        throw std::invalid_argument("CONTRACT VIOLATION: Pressure, RHS, or mask vector size mismatch.");
    }

    const double idx2 = 1.0 / (dx * dx);
    const double idy2 = 1.0 / (dy * dy);
    const double idz2 = 1.0 / (dz * dz);
    const double factor = 0.5 / (idx2 + idy2 + idz2);

    for (int iter = 0; iter < max_iters; ++iter) {
        
        // --- PASS 1: Update RED Interior Fluid Cells ---
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    if ((i + j + k) % 2 != 0) continue;
                    
                    const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    if (mask[idx] != 1) continue;

                    const size_t idx_west  = static_cast<size_t>(get_flat_index(i - 1, j, k, nx, ny));
                    const size_t idx_east  = static_cast<size_t>(get_flat_index(i + 1, j, k, nx, ny));
                    const size_t idx_south = static_cast<size_t>(get_flat_index(i, j - 1, k, nx, ny));
                    const size_t idx_north = static_cast<size_t>(get_flat_index(i, j + 1, k, nx, ny));
                    const size_t idx_down  = static_cast<size_t>(get_flat_index(i, j, k - 1, nx, ny));
                    const size_t idx_up    = static_cast<size_t>(get_flat_index(i, j, k + 1, nx, ny));

                    const double p_west  = p[idx_west];
                    const double p_east  = p[idx_east];
                    const double p_south = p[idx_south];
                    const double p_north = p[idx_north];
                    const double p_down  = p[idx_down];
                    const double p_up    = p[idx_up];

                    p[idx] = factor * (
                        (p_east + p_west) * idx2 +
                        (p_north + p_south) * idy2 +
                        (p_up + p_down) * idz2 -
                        rhs[idx]
                    );
                }
            }
        }

        // --- PASS 2: Update BLACK Interior Fluid Cells ---
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    if ((i + j + k) % 2 == 0) continue;
                    
                    const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    if (mask[idx] != 1) continue;

                    const size_t idx_west  = static_cast<size_t>(get_flat_index(i - 1, j, k, nx, ny));
                    const size_t idx_east  = static_cast<size_t>(get_flat_index(i + 1, j, k, nx, ny));
                    const size_t idx_south = static_cast<size_t>(get_flat_index(i, j - 1, k, nx, ny));
                    const size_t idx_north = static_cast<size_t>(get_flat_index(i, j + 1, k, nx, ny));
                    const size_t idx_down  = static_cast<size_t>(get_flat_index(i, j, k - 1, nx, ny));
                    const size_t idx_up    = static_cast<size_t>(get_flat_index(i, j, k + 1, nx, ny));

                    const double p_west  = p[idx_west];
                    const double p_east  = p[idx_east];
                    const double p_south = p[idx_south];
                    const double p_north = p[idx_north];
                    const double p_down  = p[idx_down];
                    const double p_up    = p[idx_up];

                    p[idx] = factor * (
                        (p_east + p_west) * idx2 +
                        (p_north + p_south) * idy2 +
                        (p_up + p_down) * idz2 -
                        rhs[idx]
                    );
                }
            }
        }

        // --- PASS 3: Synchronize Boundaries & Solids Inside Iteration ---
        for (size_t b = 0; b < bc_list.size(); ++b) {
            const auto& bc = bc_list[b];
            if (bc.type != "pressure") {
                apply_neumann_pressure(p, bc.location, nx, ny, nz, dx, dy, dz, density, gravity);
            }
        }

        apply_solid_neumann_pressure_parallel(p, mask, nx, ny, nz);
    }
}

} // namespace navier_stokes_solver
