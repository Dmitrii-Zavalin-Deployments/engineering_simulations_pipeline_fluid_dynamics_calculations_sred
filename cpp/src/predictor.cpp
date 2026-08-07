/**
 * @file predictor.cpp
 * @brief Implementation of Step 1 Predictor Trial Velocity Computation.
 */

#include "predictor.hpp"
#include "advection.hpp"
#include "laplacian.hpp"
#include <stdexcept>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ops {

namespace {

inline size_t get_index(int i, int j, int k, int ny, int nz) {
    return static_cast<size_t>(i) * (ny * nz) + static_cast<size_t>(j) * nz + k;
}

void validate_inputs(
    const GridDimensions& dims,
    const FluidProperties& fluid,
    const double* u, const double* v, const double* w,
    const double* fx, const double* fy, const double* fz,
    const double* u_star, const double* v_star, const double* w_star
) {
    if (!u || !v || !w || !fx || !fy || !fz || !u_star || !v_star || !w_star) {
        throw std::invalid_argument("CONTRACT VIOLATION: Null pointer supplied to predictor module.");
    }
    if (dims.nx < 3 || dims.ny < 3 || dims.nz < 3) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid dimensions must be at least 3x3x3 for central stencils.");
    }
    if (dims.dx <= 0.0 || dims.dy <= 0.0 || dims.dz <= 0.0) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid spacing (dx, dy, dz) must be strictly positive.");
    }
    if (fluid.dt <= 0.0) {
        throw std::invalid_argument("TEMPORAL ERROR: Time step dt must be strictly positive.");
    }
    if (fluid.nu < 0.0) {
        throw std::invalid_argument("PHYSICS ERROR: Kinematic viscosity nu cannot be negative.");
    }
}

} // anonymous namespace

void compute_trial_velocities(
    const GridDimensions& dims,
    const FluidProperties& fluid,
    const double* u, const double* v, const double* w,
    const double* fx, const double* fy, const double* fz,
    double* u_star, double* v_star, double* w_star
) {
    validate_inputs(dims, fluid, u, v, w, fx, fy, fz, u_star, v_star, w_star);

    const size_t nx = dims.nx;
    const size_t ny = dims.ny;
    const size_t nz = dims.nz;
    const size_t total_cells = nx * ny * nz;

    // Convert grid dimensions to int to match repository operator signatures
    const int Nx_int = static_cast<int>(nx);
    const int Ny_int = static_cast<int>(ny);
    const int Nz_int = static_cast<int>(nz);

    // 1. Allocate temporary field buffers for advection and Laplacian terms
    std::vector<double> adv_u(total_cells, 0.0);
    std::vector<double> adv_v(total_cells, 0.0);
    std::vector<double> adv_w(total_cells, 0.0);

    std::vector<double> lap_u(total_cells, 0.0);
    std::vector<double> lap_v(total_cells, 0.0);
    std::vector<double> lap_w(total_cells, 0.0);

    // 2. Compute domain-wide advection fields using repository operator
    compute_advection(u, v, w, u, adv_u.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_advection(u, v, w, v, adv_v.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_advection(u, v, w, w, adv_w.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);

    // 3. Compute domain-wide Laplacian fields using repository operator
    compute_laplacian(u, lap_u.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_laplacian(v, lap_v.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_laplacian(w, lap_w.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);

    // 4. Parallel Temporal Integration (Forward-Euler Predictor Step)
    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int i = 1; i < Nx_int - 1; ++i) {
        for (int j = 1; j < Ny_int - 1; ++j) {
            for (int k = 1; k < Nz_int - 1; ++k) {
                const size_t idx = get_index(i, j, k, Ny_int, Nz_int);

                double u_t = u[idx] + fluid.dt * (-adv_u[idx] + fluid.nu * lap_u[idx] + fx[idx]);
                double v_t = v[idx] + fluid.dt * (-adv_v[idx] + fluid.nu * lap_v[idx] + fy[idx]);
                double w_t = w[idx] + fluid.dt * (-adv_w[idx] + fluid.nu * lap_w[idx] + fz[idx]);

                if (!std::isfinite(u_t) || !std::isfinite(v_t) || !std::isfinite(w_t)) {
                    #pragma omp critical
                    {
                        throw std::runtime_error("MATH FAILURE: Non-finite trial velocity calculated in predictor.");
                    }
                }

                u_star[idx] = u_t;
                v_star[idx] = v_t;
                w_star[idx] = w_t;
            }
        }
    }
}

} // namespace ops
