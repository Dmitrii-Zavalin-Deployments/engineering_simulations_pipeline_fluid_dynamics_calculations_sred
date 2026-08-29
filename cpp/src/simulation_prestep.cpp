/**
 * @file simulation_prestep.cpp
 * @brief Implementation of Pre-Step Boundary & Initial Condition Setup using layered overwrite precedence, explicit mask-based wall detection, and collocated cell-center field handling.
 */

#include "orchestrator.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"
#include <stdexcept>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

inline bool is_boundary_cell(int i, int j, int k, int nx, int ny, int nz) {
    return (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1);
}

inline bool matches_location(int i, int j, int k, int nx, int ny, int nz, const std::string& location) {
    if (location == "x_min") return i == 0;
    if (location == "x_max") return i == nx - 1;
    if (location == "y_min") return j == 0;
    if (location == "y_max") return j == ny - 1;
    if (location == "z_min") return k == 0;
    if (location == "z_max") return k == nz - 1;
    if (location == "wall")  return is_boundary_cell(i, j, k, nx, ny, nz);
    return false;
}

void execute_pre_step(
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    std::vector<double>& p,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    int nx, int ny, int nz,
    bool cold_start
) {
    if (nx < 3 || ny < 3 || nz < 3) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid dimensions must be at least 3x3x3 in execute_pre_step.");
    }

    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    if (u.size() != total_cells || v.size() != total_cells || w.size() != total_cells || 
        p.size() != total_cells || mask.size() != total_cells) {
        throw std::invalid_argument("CONTRACT VIOLATION: Field vector size mismatch in execute_pre_step.");
    }

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: simulation_prestep.cpp | Operations (Cells): " << total_cells 
              << " | Grid: " << nx << "x" << ny << "x" << nz 
              << " | Active Threads: " << active_threads 
              << " | Cold Start: " << (cold_start ? "true" : "false") << "\n";

    // Uniform free-stream initialization extracted dynamically from inflow boundary conditions on cold start
    if (cold_start) {
        double init_u = 0.0;
        double init_v = 0.0;
        double init_w = 0.0;
        double init_p = 0.0;

        for (const auto& bc : bc_list) {
            if (bc.type == "inflow") {
                init_u = bc.values.u;
                init_v = bc.values.v;
                init_w = bc.values.w;
                init_p = bc.values.p;
                break;
            }
        }

        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    if (mask[idx] == 1) {
                        u[idx] = init_u;
                        v[idx] = init_v;
                        w[idx] = init_w;
                        p[idx] = init_p;
                    }
                }
            }
        }
    }

    std::vector<BoundaryCondition> wall_bc_list;
    std::vector<BoundaryCondition> face_bc_list;

    for (const auto& bc : bc_list) {
        if (bc.location == "wall") {
            wall_bc_list.push_back(bc);
        } else {
            face_bc_list.push_back(bc);
        }
    }

    auto get_interior_index = [&](int i, int j, int k) -> size_t {
        int ii = (i == 0) ? 1 : (i == nx - 1) ? nx - 2 : i;
        int jj = (j == 0) ? 1 : (j == ny - 1) ? ny - 2 : j;
        int kk = (k == 0) ? 1 : (k == nz - 1) ? nz - 2 : k;
        return static_cast<size_t>(get_flat_index(ii, jj, kk, nx, ny));
    };

    auto apply_bc = [&](const BoundaryCondition& bc, int i, int j, int k, size_t idx) {
        size_t int_idx = get_interior_index(i, j, k);

        if (bc.type == "no-slip") {
            u[idx] = bc.values.u;
            v[idx] = bc.values.v;
            w[idx] = bc.values.w;
            if (bc.values.p != 0.0) p[idx] = bc.values.p;
        } 
        else if (bc.type == "free-slip") {
            double u_new = u[idx];
            double v_new = v[idx];
            double w_new = w[idx];

            if (i == 0 || i == nx - 1) {
                u_new = 0.0;
            } else {
                u_new = (bc.values.u != 0.0) ? bc.values.u : u[int_idx];
            }

            if (j == 0 || j == ny - 1) {
                v_new = 0.0;
            } else {
                v_new = (bc.values.v != 0.0) ? bc.values.v : v[int_idx];
            }

            if (k == 0 || k == nz - 1) {
                w_new = 0.0;
            } else {
                w_new = (bc.values.w != 0.0) ? bc.values.w : w[int_idx];
            }

            u[idx] = u_new;
            v[idx] = v_new;
            w[idx] = w_new;

            if (bc.values.p != 0.0) p[idx] = bc.values.p;
        } 
        else if (bc.type == "inflow") {
            u[idx] = bc.values.u;
            v[idx] = bc.values.v;
            w[idx] = bc.values.w;
        }
        else if (bc.type == "outflow") {
            u[idx] = (bc.values.u != 0.0) ? bc.values.u : u[int_idx];
            v[idx] = (bc.values.v != 0.0) ? bc.values.v : v[int_idx];
            w[idx] = (bc.values.w != 0.0) ? bc.values.w : w[int_idx];
            p[idx] = bc.values.p;
        }
        else if (bc.type == "pressure") {
            p[idx] = bc.values.p;
        }
    };

    // Pass 1: wall BCs only on explicit wall cells (mask == -1 or 0)
    for (const auto& bc : wall_bc_list) {
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    if (mask[idx] == -1 || mask[idx] == 0) {
                        apply_bc(bc, i, j, k, idx);
                    }
                }
            }
        }
    }

    // Pass 2: face BCs, but do not overwrite explicit wall cells
    for (const auto& bc : face_bc_list) {
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (!matches_location(i, j, k, nx, ny, nz, bc.location)) {
                        continue;
                    }
                    size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    if (mask[idx] == -1 || mask[idx] == 0) {
                        continue;
                    }
                    apply_bc(bc, i, j, k, idx);
                }
            }
        }
    }
}

} // namespace navier_stokes_solver
