/**
 * @file predictor.cpp
 * @brief Implementation of Step 1 Predictor Trial Velocity Computation.
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

void validate_inputs(
    const GridDimensions& dims,
    const FluidProperties& fluid,
    double dt,
    const double* u, const double* v, const double* w,
    const double* fx, const double* fy, const double* fz,
    const std::vector<int>& mask,
    const double* u_star, const double* v_star, const double* w_star
) {
    if (!u || !v || !w || !fx || !fy || !fz || !u_star || !v_star || !w_star) {
        throw std::invalid_argument("CONTRACT VIOLATION: Null pointer supplied to predictor module.");
    }
    const size_t total_cells = dims.nx * dims.ny * dims.nz;
    if (mask.size() != total_cells) {
        throw std::invalid_argument("CONTRACT VIOLATION: Mask vector size does not match grid dimensions.");
    }
    if (dims.nx < 3 || dims.ny < 3 || dims.nz < 3) {
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
}



void compute_trial_velocities(
    const GridDimensions& dims,
    const FluidProperties& fluid,
    double dt,
    const double* u, const double* v, const double* w,
    const double* fx, const double* fy, const double* fz,
    const std::vector<int>& mask,
    double* u_star, double* v_star, double* w_star
) {
    validate_inputs(dims, fluid, dt, u, v, w, fx, fy, fz, mask, u_star, v_star, w_star);

    const size_t nx = dims.nx;
    const size_t ny = dims.ny;
    const size_t nz = dims.nz;
    const size_t total_cells = nx * ny * nz;

    const int Nx_int = static_cast<int>(nx);
    const int Ny_int = static_cast<int>(ny);
    const int Nz_int = static_cast<int>(nz);

    // 1. Copy current state to star fields as baseline.
    // This automatically preserves all pre-step Dirichlet boundary values (mask == -1) 
    // and solid states (mask == 0) without corruption or uninitialized garbage.
    std::copy(u, u + total_cells, u_star);
    std::copy(v, v + total_cells, v_star);
    std::copy(w, w + total_cells, w_star);

    // 2. Allocate temporary field buffers for advection and Laplacian terms
    std::vector<double> adv_u(total_cells, 0.0);
    std::vector<double> adv_v(total_cells, 0.0);
    std::vector<double> adv_w(total_cells, 0.0);

    std::vector<double> lap_u(total_cells, 0.0);
    std::vector<double> lap_v(total_cells, 0.0);
    std::vector<double> lap_w(total_cells, 0.0);

    // 3. Compute domain-wide advection fields using repository operators
    compute_advection(u, v, w, u, adv_u.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_advection(u, v, w, v, adv_v.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_advection(u, v, w, w, adv_w.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);

    // 4. Compute domain-wide Laplacian fields using repository operators
    compute_laplacian(u, lap_u.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_laplacian(v, lap_v.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_laplacian(w, lap_w.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);

    // 5. Parallel Temporal Integration (Forward-Euler Predictor Step)
    // Executed STRICTLY on active fluid cells (mask == 1) to respect physical constraints.
    bool has_non_finite = false;

    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000) reduction(||:has_non_finite)
    for (int i = 0; i < Nx_int; ++i) {
        for (int j = 0; j < Ny_int; ++j) {
            for (int k = 0; k < Nz_int; ++k) {
                const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, Nx_int, Ny_int));

                if (mask[idx] != 1) continue; // Skip non-fluid cells (boundaries and solids)

                double u_t = u[idx] + dt * (-adv_u[idx] + fluid.nu * lap_u[idx] + fx[idx] / fluid.density);
                double v_t = v[idx] + dt * (-adv_v[idx] + fluid.nu * lap_v[idx] + fy[idx] / fluid.density);
                double w_t = w[idx] + dt * (-adv_w[idx] + fluid.nu * lap_w[idx] + fz[idx] / fluid.density);

                if (!std::isfinite(u_t) || !std::isfinite(v_t) || !std::isfinite(w_t)) {
                    has_non_finite = true;
                }

                u_star[idx] = u_t;
                v_star[idx] = v_t;
                w_star[idx] = w_t;
            }
        }
    }

    if (has_non_finite) {
        throw std::runtime_error("MATH FAILURE: Non-finite trial velocity calculated in predictor.");
    }
}

} // namespace navier_stokes_solver
