/**
 * @file pressure_poisson_solver.cpp
 * @brief Implementation of Step 3 Pressure Poisson Solver (Red-Black GS) with robust safety validation,
 *        hydrostatic pressure / body-force boundary balancing, and execution tracing.
 */

#include "pressure_poisson_solver.hpp"
#include "grid_math.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void apply_neumann_pressure(
    std::vector<double>& p,
    const std::string& location,
    const DirichletFaces& dirichlet,
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

    const double dp_dx = density * gx;
    const double dp_dy = density * gy;
    const double dp_dz = density * gz;

    std::vector<double> p_tmp = p;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int raw_idx = get_flat_index(i, j, k, nx, ny);
                if (raw_idx < 0) continue;
                const size_t idx = static_cast<size_t>(raw_idx);

                int count = 0;
                double val = 0.0;

                // Accumulate normal pressure gradient conditions across active boundary faces, skipping Dirichlet anchors
                if ((location == "x_min" || location == "wall") && i == 0 && !dirichlet.x_min && nx > 1) {
                    const int neighbor = get_flat_index(1, j, k, nx, ny);
                    if (neighbor >= 0) {
                        val += p[static_cast<size_t>(neighbor)] - dp_dx * dx;
                        count++;
                    }
                }
                if ((location == "x_max" || location == "wall") && i == nx - 1 && !dirichlet.x_max && nx > 1) {
                    const int neighbor = get_flat_index(nx - 2, j, k, nx, ny);
                    if (neighbor >= 0) {
                        val += p[static_cast<size_t>(neighbor)] + dp_dx * dx;
                        count++;
                    }
                }
                if ((location == "y_min" || location == "wall") && j == 0 && !dirichlet.y_min && ny > 1) {
                    const int neighbor = get_flat_index(i, 1, k, nx, ny);
                    if (neighbor >= 0) {
                        val += p[static_cast<size_t>(neighbor)] - dp_dy * dy;
                        count++;
                    }
                }
                if ((location == "y_max" || location == "wall") && j == ny - 1 && !dirichlet.y_max && ny > 1) {
                    const int neighbor = get_flat_index(i, ny - 2, k, nx, ny);
                    if (neighbor >= 0) {
                        val += p[static_cast<size_t>(neighbor)] + dp_dy * dy;
                        count++;
                    }
                }
                if ((location == "z_min" || location == "wall") && k == 0 && !dirichlet.z_min && nz > 1) {
                    const int neighbor = get_flat_index(i, j, 1, nx, ny);
                    if (neighbor >= 0) {
                        val += p[static_cast<size_t>(neighbor)] - dp_dz * dz;
                        count++;
                    }
                }
                if ((location == "z_max" || location == "wall") && k == nz - 1 && !dirichlet.z_max && nz > 1) {
                    const int neighbor = get_flat_index(i, j, nz - 2, nx, ny);
                    if (neighbor >= 0) {
                        val += p[static_cast<size_t>(neighbor)] + dp_dz * dz;
                        count++;
                    }
                }

                // If cell is on one or more boundaries, average the target boundary values
                if (count > 0) {
                    p_tmp[idx] = val / static_cast<double>(count);
                }
            }
        }
    }
    p = std::move(p_tmp);
}

void apply_solid_neumann_pressure_parallel(
    std::vector<double>& p, 
    const std::vector<int>& mask, 
    int nx, int ny, int nz,
    double dx, double dy, double dz
) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) return;

    std::vector<double> p_tmp = p;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int raw_idx = get_flat_index(i, j, k, nx, ny);
                if (raw_idx < 0) continue;
                const size_t idx = static_cast<size_t>(raw_idx);

                if (mask[idx] != 0) continue; // Restrict strictly to internal solid cells (mask == 0), leaving outer wall cells intact

                int count = 0;
                double val = 0.0;

                // Accumulate pressures from valid adjacent active fluid cells with boundary guards
                if (i > 0) {
                    const int idx_west = get_flat_index(i - 1, j, k, nx, ny);
                    if (idx_west >= 0 && mask[static_cast<size_t>(idx_west)] == 1) {
                        val += p[static_cast<size_t>(idx_west)];
                        count++;
                    }
                }
                if (i < nx - 1) {
                    const int idx_east = get_flat_index(i + 1, j, k, nx, ny);
                    if (idx_east >= 0 && mask[static_cast<size_t>(idx_east)] == 1) {
                        val += p[static_cast<size_t>(idx_east)];
                        count++;
                    }
                }
                if (j > 0) {
                    const int idx_south = get_flat_index(i, j - 1, k, nx, ny);
                    if (idx_south >= 0 && mask[static_cast<size_t>(idx_south)] == 1) {
                        val += p[static_cast<size_t>(idx_south)];
                        count++;
                    }
                }
                if (j < ny - 1) {
                    const int idx_north = get_flat_index(i, j + 1, k, nx, ny);
                    if (idx_north >= 0 && mask[static_cast<size_t>(idx_north)] == 1) {
                        val += p[static_cast<size_t>(idx_north)];
                        count++;
                    }
                }
                if (k > 0) {
                    const int idx_down = get_flat_index(i, j, k - 1, nx, ny);
                    if (idx_down >= 0 && mask[static_cast<size_t>(idx_down)] == 1) {
                        val += p[static_cast<size_t>(idx_down)];
                        count++;
                    }
                }
                if (k < nz - 1) {
                    const int idx_up = get_flat_index(i, j, k + 1, nx, ny);
                    if (idx_up >= 0 && mask[static_cast<size_t>(idx_up)] == 1) {
                        val += p[static_cast<size_t>(idx_up)];
                        count++;
                    }
                }

                if (count > 0) {
                    p_tmp[idx] = val / static_cast<double>(count);
                }
            }
        }
    }
    p = std::move(p_tmp);
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

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: pressure_poisson_solver.cpp | Operations (Cells): " << total_cells 
              << " | Grid: " << nx << "x" << ny << "x" << nz 
              << " | Active Threads: " << active_threads << "\n";

    // Map out domain faces anchored by fixed pressure or outflow back-pressure conditions
    DirichletFaces dirichlet;
    for (const auto& bc : bc_list) {
        if (bc.type == "pressure" || bc.type == "outflow") {
            if (bc.location == "x_min") dirichlet.x_min = true;
            if (bc.location == "x_max") dirichlet.x_max = true;
            if (bc.location == "y_min") dirichlet.y_min = true;
            if (bc.location == "y_max") dirichlet.y_max = true;
            if (bc.location == "z_min") dirichlet.z_min = true;
            if (bc.location == "z_max") dirichlet.z_max = true;
        }
    }

    const double idx2 = 1.0 / (dx * dx);
    const double idy2 = 1.0 / (dy * dy);
    const double idz2 = 1.0 / (dz * dz);
    const double factor = 0.5 / (idx2 + idy2 + idz2);

    bool has_error = false;
    int err_i = 0, err_j = 0, err_k = 0;
    double err_val = 0.0;

    for (int iter = 0; iter < max_iters; ++iter) {
        
        // --- PASS 1: Update RED Interior Fluid Cells ((i + j + k) % 2 == 0) ---
        #pragma omp parallel for collapse(3) schedule(static) if(total_cells >= 1000)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    if ((i + j + k) % 2 != 0) continue;

                    const int raw_idx = get_flat_index(i, j, k, nx, ny);
                    if (raw_idx < 0) continue;
                    const size_t idx = static_cast<size_t>(raw_idx);

                    if (mask[idx] != 1) continue;

                    const int w = get_flat_index(i - 1, j, k, nx, ny);
                    const int e = get_flat_index(i + 1, j, k, nx, ny);
                    const int s = get_flat_index(i, j - 1, k, nx, ny);
                    const int n = get_flat_index(i, j + 1, k, nx, ny);
                    const int d = get_flat_index(i, j, k - 1, nx, ny);
                    const int u = get_flat_index(i, j, k + 1, nx, ny);

                    if (w < 0 || e < 0 || s < 0 || n < 0 || d < 0 || u < 0) continue;

                    const size_t idx_west  = static_cast<size_t>(w);
                    const size_t idx_east  = static_cast<size_t>(e);
                    const size_t idx_south = static_cast<size_t>(s);
                    const size_t idx_north = static_cast<size_t>(n);
                    const size_t idx_down  = static_cast<size_t>(d);
                    const size_t idx_up    = static_cast<size_t>(u);

                    // Mask-aware neighbor pressure evaluation (enforces Neumann dp/dn = 0 at boundaries/solids)
                    const double p_west  = (mask[idx_west] == 1)  ? p[idx_west]  : p[idx];
                    const double p_east  = (mask[idx_east] == 1)  ? p[idx_east]  : p[idx];
                    const double p_south = (mask[idx_south] == 1) ? p[idx_south] : p[idx];
                    const double p_north = (mask[idx_north] == 1) ? p[idx_north] : p[idx];
                    const double p_down  = (mask[idx_down] == 1)  ? p[idx_down]  : p[idx];
                    const double p_up    = (mask[idx_up] == 1)    ? p[idx_up]    : p[idx];

                    double p_new = factor * (
                        (p_east + p_west) * idx2 +
                        (p_north + p_south) * idy2 +
                        (p_up + p_down) * idz2 -
                        rhs[idx]
                    );

                    if (!std::isfinite(p_new)) {
                        #pragma omp critical
                        {
                            if (!has_error) {
                                has_error = true;
                                err_i = i;
                                err_j = j;
                                err_k = k;
                                err_val = p_new;
                            }
                        }
                    }

                    p[idx] = p_new;
                }
            }
        }

        // --- PASS 2: Update BLACK Interior Fluid Cells ((i + j + k) % 2 != 0) ---
        #pragma omp parallel for collapse(3) schedule(static) if(total_cells >= 1000)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    if ((i + j + k) % 2 == 0) continue;

                    const int raw_idx = get_flat_index(i, j, k, nx, ny);
                    if (raw_idx < 0) continue;
                    const size_t idx = static_cast<size_t>(raw_idx);

                    if (mask[idx] != 1) continue;

                    const int w = get_flat_index(i - 1, j, k, nx, ny);
                    const int e = get_flat_index(i + 1, j, k, nx, ny);
                    const int s = get_flat_index(i, j - 1, k, nx, ny);
                    const int n = get_flat_index(i, j + 1, k, nx, ny);
                    const int d = get_flat_index(i, j, k - 1, nx, ny);
                    const int u = get_flat_index(i, j, k + 1, nx, ny);

                    if (w < 0 || e < 0 || s < 0 || n < 0 || d < 0 || u < 0) continue;

                    const size_t idx_west  = static_cast<size_t>(w);
                    const size_t idx_east  = static_cast<size_t>(e);
                    const size_t idx_south = static_cast<size_t>(s);
                    const size_t idx_north = static_cast<size_t>(n);
                    const size_t idx_down  = static_cast<size_t>(d);
                    const size_t idx_up    = static_cast<size_t>(u);

                    // Mask-aware neighbor pressure evaluation (enforces Neumann dp/dn = 0 at boundaries/solids)
                    const double p_west  = (mask[idx_west] == 1)  ? p[idx_west]  : p[idx];
                    const double p_east  = (mask[idx_east] == 1)  ? p[idx_east]  : p[idx];
                    const double p_south = (mask[idx_south] == 1) ? p[idx_south] : p[idx];
                    const double p_north = (mask[idx_north] == 1) ? p[idx_north] : p[idx];
                    const double p_down  = (mask[idx_down] == 1)  ? p[idx_down]  : p[idx];
                    const double p_up    = (mask[idx_up] == 1)    ? p[idx_up]    : p[idx];

                    double p_new = factor * (
                        (p_east + p_west) * idx2 +
                        (p_north + p_south) * idy2 +
                        (p_up + p_down) * idz2 -
                        rhs[idx]
                    );

                    if (!std::isfinite(p_new)) {
                        #pragma omp critical
                        {
                            if (!has_error) {
                                has_error = true;
                                err_i = i;
                                err_j = j;
                                err_k = k;
                                err_val = p_new;
                            }
                        }
                    }

                    p[idx] = p_new;
                }
            }
        }

        if (has_error) {
            std::cerr << "MATH FAILURE: Non-finite pressure detected at grid index [" 
                      << err_i << ", " << err_j << ", " << err_k << "] | Result: " << err_val << "\n";
            throw std::runtime_error("Pressure Poisson solver exploded. Pressure field is non-finite.");
        }

        // --- PASS 3: Synchronize Boundaries & Solids Inside Iteration ---
        for (size_t b = 0; b < bc_list.size(); ++b) {
            const auto& bc = bc_list[b];
            if (bc.type != "pressure" && bc.type != "outflow") {
                apply_neumann_pressure(p, bc.location, dirichlet, nx, ny, nz, dx, dy, dz, density, gravity);
            }
        }

        apply_solid_neumann_pressure_parallel(p, mask, nx, ny, nz, dx, dy, dz);
    }
}

} // namespace navier_stokes_solver

