/**
 * @file predictor.hpp
 * @brief Step 1 Predictor Phase Orchestrator for Incompressible Navier-Stokes.
 */

#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include <cstddef>
#include <vector>
#include "orchestrator.hpp"

namespace navier_stokes_solver {

struct FluidProperties {
    double nu;
    double density;
};

/**
 * @brief Computes the trial velocity vector field (u*, v*, w*) using explicit temporal integration.
 *        Applies updates strictly to active fluid cells (mask == 1), while preserving 
 *        pre-step boundary and solid states.
 */
void compute_trial_velocities(
    const GridDimensions& dims,
    const FluidProperties& fluid,
    double dt,
    const double* u, const double* v, const double* w,
    const double* fx, const double* fy, const double* fz,
    const std::vector<int>& mask,
    double* u_star, double* v_star, double* w_star
);

} // namespace navier_stokes_solver

#endif // PREDICTOR_HPP
