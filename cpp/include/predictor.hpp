/**
 * @file predictor.hpp
 * @brief Step 1 Predictor Phase Orchestrator for Incompressible Navier-Stokes.
 */

#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include <cstddef>

namespace ops {

struct GridDimensions {
    size_t nx;
    size_t ny;
    size_t nz;
    double dx;
    double dy;
    double dz;
};

struct FluidProperties {
    double dt;
    double nu;
};

/**
 * @brief Computes the trial velocity vector field (u*, v*, w*) using explicit temporal integration.
 */
void compute_trial_velocities(
    const GridDimensions& dims,
    const FluidProperties& fluid,
    const double* u, const double* v, const double* w,
    const double* fx, const double* fy, const double* fz,
    double* u_star, double* v_star, double* w_star
);

} // namespace ops

#endif // PREDICTOR_HPP
