/**
 * @file predictor.cpp
 * @brief Implementation of Step 1 Predictor Trial Velocity Computation.
 */

#include "predictor.hpp"
#include "advection.hpp"
#include "laplacian.hpp"
#include <stdexcept>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ops {

namespace {

inline size_t get_index(size_t i, size_t j, size_t k, size_t ny, size_t nz) {
    return i * (ny * nz) + j * nz + k;
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

    #pragma omp parallel for collapse(2) schedule(static) if(nx * ny * nz > 1000)
    for (size_t i = 1; i < nx - 1; ++i) {
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t k = 1; k < nz - 1; ++k) {
                const size_t idx = get_index(i, j, k, ny, nz);

                double adv_x = compute_advection_x(u, v, w, i, j, k, dims.dx, dims.dy, dims.dz, ny, nz);
                double adv_y = compute_advection_y(u, v, w, i, j, k, dims.dx, dims.dy, dims.dz, ny, nz);
                double adv_z = compute_advection_z(u, v, w, i, j, k, dims.dx, dims.dy, dims.dz, ny, nz);

                double lap_u = compute_laplacian(u, i, j, k, dims.dx, dims.dy, dims.dz, ny, nz);
                double lap_v = compute_laplacian(v, i, j, k, dims.dx, dims.dy, dims.dz, ny, nz);
                double lap_w = compute_laplacian(w, i, j, k, dims.dx, dims.dy, dims.dz, ny, nz);

                double u_t = u[idx] + fluid.dt * (-adv_x + fluid.nu * lap_u + fx[idx]);
                double v_t = v[idx] + fluid.dt * (-adv_y + fluid.nu * lap_v + fy[idx]);
                double w_t = w[idx] + fluid.dt * (-adv_z + fluid.nu * lap_w + fz[idx]);

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
