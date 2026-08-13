/**
 * @file predictor.cpp
 * @brief Implementation of Step 1 Predictor Trial Velocity Computation for staggered MAC grid with 3D Gravity Integration.
 */

#include "predictor.hpp"
#include "advection.hpp"
#include "laplacian.hpp"
#include "grid_math.hpp"
#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void compute_trial_velocities(
    const GridDimensions& dims,
    const FluidProperties& fluid,
    double dt,
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    const std::vector<double>& fx,
    const std::vector<double>& fy,
    const std::vector<double>& fz,
    const std::vector<double>& gravity,
    const std::vector<int>& mask,
    std::vector<double>& u_star,
    std::vector<double>& v_star,
    std::vector<double>& w_star
) {
    const int nx = dims.nx;
    const int ny = dims.ny;
    const int nz = dims.nz;

    if (nx < 3 || ny < 3 || nz < 3) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid dimensions must be at least 3x3x3 for central stencils.");
    }
    if (dims.dx <= 0.0 || dims.dy <= 0.0 || dims.dz <= 0.0) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid spacing (dx, dy, dz) must be strictly positive.");
    }
    if (dt <= 0.0) {
        throw std::invalid_argument("TEMPORAL ERROR: Time step dt must be strictly positive.");
    }
    if (fluid.nu < 0.0) {
        throw std::invalid_argument("PHYSICS ERROR: Kinematic viscosity nu cannot be negative.");
    }
    if (fluid.density <= 0.0) {
        throw std::invalid_argument("PHYSICS ERROR: Fluid density must be strictly positive.");
    }
    if (gravity.size() != 3) {
        throw std::invalid_argument("CONTRACT VIOLATION: gravity vector must contain exactly 3 components [gx, gy, gz].");
    }

    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    const size_t u_size = static_cast<size_t>(nx + 1) * ny * nz;
    const size_t v_size = static_cast<size_t>(nx) * (ny + 1) * nz;
    const size_t w_size = static_cast<size_t>(nx) * ny * (nz + 1);

    if (mask.size() != total_cells || fx.size() != total_cells || fy.size() != total_cells || fz.size() != total_cells) {
        throw std::invalid_argument("CONTRACT VIOLATION: Cell-centered vector size mismatch in predictor module.");
    }
    if (u.size() != u_size || v.size() != v_size || w.size() != w_size ||
        u_star.size() != u_size || v_star.size() != v_size || w_star.size() != w_size) {
        throw std::invalid_argument("CONTRACT VIOLATION: Staggered velocity vector size mismatch in predictor module.");
    }

    // 1. Copy current state to star fields as baseline.
    // Preserves Dirichlet boundary values and solid states without corruption.
    u_star = u;
    v_star = v;
    w_star = w;

    bool has_non_finite = false;
    const double gx = gravity[0];
    const double gy = gravity[1];
    const double gz = gravity[2];

    // 2. Update u-velocities at x-faces (i from 1 to nx - 1)
    #pragma omp parallel for collapse(3) schedule(static) reduction(||:has_non_finite)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                size_t u_idx = static_cast<size_t>(i) + j * (nx + 1) + k * (nx + 1) * ny;
                size_t p_left  = static_cast<size_t>(i - 1) + j * nx + k * nx * ny;
                size_t p_right = static_cast<size_t>(i)     + j * nx + k * nx * ny;

                if (mask[p_left] == 1 && mask[p_right] == 1) {
                    double force_x = 0.5 * (fx[p_left] + fx[p_right]);
                    double u_val = u[u_idx];
                    double u_t = u_val + dt * (force_x / fluid.density + gx);

                    if (!std::isfinite(u_t)) {
                        has_non_finite = true;
                    }
                    u_star[u_idx] = u_t;
                }
            }
        }
    }

    // 3. Update v-velocities at y-faces (j from 1 to ny - 1)
    #pragma omp parallel for collapse(3) schedule(static) reduction(||:has_non_finite)
    for (int k = 0; k < nz; ++k) {
        for (int j = 1; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t v_idx = static_cast<size_t>(i) + j * nx + k * nx * (ny + 1);
                size_t p_south = static_cast<size_t>(i) + (j - 1) * nx + k * nx * ny;
                size_t p_north = static_cast<size_t>(i) + j * nx + k * nx * ny;

                if (mask[p_south] == 1 && mask[p_north] == 1) {
                    double force_y = 0.5 * (fy[p_south] + fy[p_north]);
                    double v_val = v[v_idx];
                    double v_t = v_val + dt * (force_y / fluid.density + gy);

                    if (!std::isfinite(v_t)) {
                        has_non_finite = true;
                    }
                    v_star[v_idx] = v_t;
                }
            }
        }
    }

    // 4. Update w-velocities at z-faces (k from 1 to nz - 1)
    #pragma omp parallel for collapse(3) schedule(static) reduction(||:has_non_finite)
    for (int k = 1; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t w_idx = static_cast<size_t>(i) + j * nx + k * nx * ny;
                size_t p_down = static_cast<size_t>(i) + j * nx + (k - 1) * nx * ny;
                size_t p_up   = static_cast<size_t>(i) + j * nx + k * nx * ny;

                if (mask[p_down] == 1 && mask[p_up] == 1) {
                    double force_z = 0.5 * (fz[p_down] + fz[p_up]);
                    double w_val = w[w_idx];
                    double w_t = w_val + dt * (force_z / fluid.density + gz);

                    if (!std::isfinite(w_t)) {
                        has_non_finite = true;
                    }
                    w_star[w_idx] = w_t;
                }
            }
        }
    }

    if (has_non_finite) {
        throw std::runtime_error("MATH FAILURE: Non-finite trial velocity calculated in predictor.");
    }
}

} // namespace navier_stokes_solver
