/**
 * @file predictor.hpp
 * @brief Header for Step 2 Predictor Trial Velocity computation with staggered MAC grid and gravity support.
 */

#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include <vector>
#include "grid_math.hpp"

namespace navier_stokes_solver {

struct FluidProperties {
    double nu;
    double density;
};

/**
 * @brief Computes the trial velocity vector field (u*, v*, w*) on a staggered MAC grid using explicit temporal integration,
 *        including external body forces and gravity vector components.
 *        Applies updates strictly to active fluid cells/faces (mask == 1), while preserving 
 *        pre-step boundary and solid states.
 */
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
);

} // namespace navier_stokes_solver

#endif // PREDICTOR_HPP
