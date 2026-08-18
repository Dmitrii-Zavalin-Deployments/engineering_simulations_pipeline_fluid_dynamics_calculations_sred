/**
 * @file simulation_prestep.cpp
 * @brief Implementation of Pre-Step Boundary & Initial Condition Setup with robust safety validation and non-overwriting exclusivity policy.
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

inline bool matches_location(int i, int j, int k, int nx, int ny, int nz, const std::string& location) {
    if (location == "x_min" && i == 0) return true;
    if (location == "x_max" && i == nx - 1) return true;
    if (location == "y_min" && j == 0) return true;
    if (location == "y_max" && j == ny - 1) return true;
    if (location == "z_min" && k == 0) return true;
    if (location == "z_max" && k == nz - 1) return true;
    if (location == "wall") {
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
            return true;
        }
    }
    return false;
}

void execute_pre_step(
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    std::vector<double>& p,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    int nx, int ny, int nz
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
              << " | Active Threads: " << active_threads << "\n";

    // Track which boundary cells have been claimed by explicit face-specific boundary conditions
    std::vector<bool> claimed_boundary(total_cells, false);

    // Separate boundary conditions into explicit face rules and generic wall rules
    // to enforce strict non-overwriting precedence.
    std::vector<BoundaryCondition> explicit_bc_list;
    std::vector<BoundaryCondition> wall_bc_list;

    for (const auto& bc : bc_list) {
        if (bc.location == "wall") {
            wall_bc_list.push_back(bc);
        } else {
            explicit_bc_list.push_back(bc);
        }
    }

    // Helper lambda to apply a boundary condition to a specific cell index
    auto apply_bc = [&](const BoundaryCondition& bc, int i, int j, int k, size_t idx) {
        if (bc.type == "inflow") {
            u[idx] = bc.u_val;
            v[idx] = bc.v_val;
            w[idx] = bc.w_val;
        } 
        else if (bc.type == "pressure" || bc.type == "outflow") {
            if (bc.type == "pressure") {
                p[idx] = bc.scalar_p;
            }

            // Zero-gradient Neumann velocity extrapolation across domain boundaries and generic walls
            if (bc.location == "x_max" || (bc.location == "wall" && i == nx - 1)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(nx - 2, j, k, nx, ny));
                u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
            }
            if (bc.location == "x_min" || (bc.location == "wall" && i == 0)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(1, j, k, nx, ny));
                u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
            }
            if (bc.location == "y_max" || (bc.location == "wall" && j == ny - 1)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(i, ny - 2, k, nx, ny));
                u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
            }
            if (bc.location == "y_min" || (bc.location == "wall" && j == 0)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(i, 1, k, nx, ny));
                u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
            }
            if (bc.location == "z_max" || (bc.location == "wall" && k == nz - 1)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(i, j, nz - 2, nx, ny));
                u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
            }
            if (bc.location == "z_min" || (bc.location == "wall" && k == 0)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(i, j, 1, nx, ny));
                u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
            }
        } 
        else if (bc.type == "no-slip") {
            u[idx] = 0.0;
            v[idx] = 0.0;
            w[idx] = 0.0;
        } 
        else if (bc.type == "free-slip") {
            if (bc.location == "x_max" || (bc.location == "wall" && i == nx - 1)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(nx - 2, j, k, nx, ny));
                u[idx] = 0.0; v[idx] = v[int_idx]; w[idx] = w[int_idx];
            }
            if (bc.location == "x_min" || (bc.location == "wall" && i == 0)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(1, j, k, nx, ny));
                u[idx] = 0.0; v[idx] = v[int_idx]; w[idx] = w[int_idx];
            }
            if (bc.location == "y_max" || (bc.location == "wall" && j == ny - 1)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(i, ny - 2, k, nx, ny));
                u[idx] = u[int_idx]; v[idx] = 0.0; w[idx] = w[int_idx];
            }
            if (bc.location == "y_min" || (bc.location == "wall" && j == 0)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(i, 1, k, nx, ny));
                u[idx] = u[int_idx]; v[idx] = 0.0; w[idx] = w[int_idx];
            }
            if (bc.location == "z_max" || (bc.location == "wall" && k == nz - 1)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(i, j, nz - 2, nx, ny));
                u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = 0.0;
            }
            if (bc.location == "z_min" || (bc.location == "wall" && k == 0)) {
                size_t int_idx = static_cast<size_t>(get_flat_index(i, j, 1, nx, ny));
                u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = 0.0;
            }
        }
    };

    // Pass 1: Apply explicit face-specific boundary conditions and mark cells as claimed
    for (size_t b = 0; b < explicit_bc_list.size(); ++b) {
        const auto& bc = explicit_bc_list[b];

        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (!matches_location(i, j, k, nx, ny, nz, bc.location)) {
                        continue;
                    }
                    const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    
                    apply_bc(bc, i, j, k, idx);

                    #pragma omp atomic write
                    claimed_boundary[idx] = true;
                }
            }
        }
    }

    // Pass 2: Apply generic wall boundary conditions ONLY to unclaimed boundary cells
    for (size_t b = 0; b < wall_bc_list.size(); ++b) {
        const auto& bc = wall_bc_list[b];

        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (!matches_location(i, j, k, nx, ny, nz, bc.location)) {
                        continue;
                    }
                    const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));

                    bool already_claimed = false;
                    #pragma omp atomic read
                    already_claimed = claimed_boundary[idx];

                    if (!already_claimed) {
                        apply_bc(bc, i, j, k, idx);
                        #pragma omp atomic write
                        claimed_boundary[idx] = true;
                    }
                }
            }
        }
    }
}

} // namespace navier_stokes_solver
