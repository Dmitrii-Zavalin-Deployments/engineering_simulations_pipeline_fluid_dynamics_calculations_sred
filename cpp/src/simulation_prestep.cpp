/**
 * @file simulation_prestep.cpp
 * @brief Implementation of Pre-Step Boundary & Initial Condition Setup using layered overwrite precedence and explicit mask-based wall detection.
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

    // Separate generic wall baseline definitions from explicit face boundary conditions
    std::vector<BoundaryCondition> wall_bc_list;
    std::vector<BoundaryCondition> face_bc_list;

    for (const auto& bc : bc_list) {
        if (bc.location == "wall") {
            wall_bc_list.push_back(bc);
        } else {
            face_bc_list.push_back(bc);
        }
    }

    // Fetch nearest interior neighbor cell index for zero-gradient Neumann extrapolation fallback
    auto get_interior_index = [&](int i, int j, int k) -> size_t {
        int ii = (i == 0) ? 1 : (i == nx - 1) ? nx - 2 : i;
        int jj = (j == 0) ? 1 : (j == ny - 1) ? ny - 2 : j;
        int kk = (k == 0) ? 1 : (k == nz - 1) ? nz - 2 : k;
        return static_cast<size_t>(get_flat_index(ii, jj, kk, nx, ny));
    };

    // Flexible boundary condition application closure
    auto apply_bc = [&](const BoundaryCondition& bc, int i, int j, int k, size_t idx) {
        size_t int_idx = get_interior_index(i, j, k);

        if (bc.type == "no-slip") {
            u[idx] = bc.u_val;
            v[idx] = bc.v_val;
            w[idx] = bc.w_val;
            if (bc.scalar_p != 0.0) p[idx] = bc.scalar_p;
        } 
        else if (bc.type == "free-slip") {
            u[idx] = (i == 0 || i == nx - 1) ? 0.0 : ((bc.u_val != 0.0) ? bc.u_val : u[int_idx]);
            v[idx] = (j == 0 || j == ny - 1) ? 0.0 : ((bc.v_val != 0.0) ? bc.v_val : v[int_idx]);
            w[idx] = (k == 0 || k == nz - 1) ? 0.0 : ((bc.w_val != 0.0) ? bc.w_val : w[int_idx]);
            if (bc.scalar_p != 0.0) p[idx] = bc.scalar_p;
        } 
        else if (bc.type == "inflow" || bc.type == "outflow" || bc.type == "pressure") {
            u[idx] = (bc.u_val != 0.0 || bc.type == "inflow") ? bc.u_val : u[int_idx];
            v[idx] = (bc.v_val != 0.0 || bc.type == "inflow") ? bc.v_val : v[int_idx];
            w[idx] = (bc.w_val != 0.0 || bc.type == "inflow") ? bc.w_val : w[int_idx];
            p[idx] = bc.scalar_p;
        }
    };

    // Pass 1: Apply generic wall boundary conditions strictly to explicit wall cells (mask == -1)
    for (const auto& bc : wall_bc_list) {
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    if (mask[idx] == -1) {
                        apply_bc(bc, i, j, k, idx);
                    }
                }
            }
        }
    }

    // Pass 2: Overwrite domain face boundary cells with explicit conditions (x_min, x_max, etc.)
    for (const auto& bc : face_bc_list) {
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (matches_location(i, j, k, nx, ny, nz, bc.location)) {
                        size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                        apply_bc(bc, i, j, k, idx);
                    }
                }
            }
        }
    }
}

} // namespace navier_stokes_solver
