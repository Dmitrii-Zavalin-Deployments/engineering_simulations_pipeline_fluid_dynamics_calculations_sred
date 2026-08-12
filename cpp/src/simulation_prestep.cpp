/**
 * @file simulation_prestep.cpp
 * @brief Implementation of Pre-Step Boundary & Initial Condition Setup with robust safety validation.
 */

#include "orchestrator.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"
#include <stdexcept>

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
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("GEOMETRY CRASH: Invalid grid dimensions in execute_pre_step.");
    }

    // Iterate through all boundary configurations defined in the input schema array
    for (size_t b = 0; b < bc_list.size(); ++b) {
        const auto& bc = bc_list[b];

        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));

                    // Check if current cell matches the target boundary location
                    if (!matches_location(i, j, k, nx, ny, nz, bc.location)) {
                        continue;
                    }

                    // Apply boundary values strictly matching schema enum types:
                    // "inflow", "outflow", "pressure", "no-slip", "free-slip"
                    if (bc.type == "inflow") {
                        u[idx] = bc.u_val;
                        v[idx] = bc.v_val;
                        w[idx] = bc.w_val;
                    } 
                    else if (bc.type == "pressure" || bc.type == "outflow") {
                        if (bc.type == "pressure") {
                            p[idx] = bc.scalar_p;
                        }

                        // Zero-gradient Neumann velocity extrapolation across domain boundaries
                        if (bc.location == "x_max" && i == nx - 1) {
                            size_t int_idx = get_flat_index(nx - 2, j, k, nx, ny);
                            u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
                        } else if (bc.location == "x_min" && i == 0) {
                            size_t int_idx = get_flat_index(1, j, k, nx, ny);
                            u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
                        } else if (bc.location == "y_max" && j == ny - 1) {
                            size_t int_idx = get_flat_index(i, ny - 2, k, nx, ny);
                            u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
                        } else if (bc.location == "y_min" && j == 0) {
                            size_t int_idx = get_flat_index(i, 1, k, nx, ny);
                            u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
                        } else if (bc.location == "z_max" && k == nz - 1) {
                            size_t int_idx = get_flat_index(i, j, nz - 2, nx, ny);
                            u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
                        } else if (bc.location == "z_min" && k == 0) {
                            size_t int_idx = get_flat_index(i, j, 1, nx, ny);
                            u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = w[int_idx];
                        }
                    } 
                    else if (bc.type == "no-slip") {
                        u[idx] = 0.0;
                        v[idx] = 0.0;
                        w[idx] = 0.0;
                    } 
                    else if (bc.type == "free-slip") {
                        if (bc.location == "x_max" && i == nx - 1) {
                            size_t int_idx = get_flat_index(nx - 2, j, k, nx, ny);
                            u[idx] = 0.0; v[idx] = v[int_idx]; w[idx] = w[int_idx];
                        } else if (bc.location == "x_min" && i == 0) {
                            size_t int_idx = get_flat_index(1, j, k, nx, ny);
                            u[idx] = 0.0; v[idx] = v[int_idx]; w[idx] = w[int_idx];
                        } else if (bc.location == "y_max" && j == ny - 1) {
                            size_t int_idx = get_flat_index(i, ny - 2, k, nx, ny);
                            u[idx] = u[int_idx]; v[idx] = 0.0; w[idx] = w[int_idx];
                        } else if (bc.location == "y_min" && j == 0) {
                            size_t int_idx = get_flat_index(i, 1, k, nx, ny);
                            u[idx] = u[int_idx]; v[idx] = 0.0; w[idx] = w[int_idx];
                        } else if (bc.location == "z_max" && k == nz - 1) {
                            size_t int_idx = get_flat_index(i, j, nz - 2, nx, ny);
                            u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = 0.0;
                        } else if (bc.location == "z_min" && k == 0) {
                            size_t int_idx = get_flat_index(i, j, 1, nx, ny);
                            u[idx] = u[int_idx]; v[idx] = v[int_idx]; w[idx] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

} // namespace navier_stokes_solver
