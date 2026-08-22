/**
 * @file predictor.hpp
 * @brief Header for Step 2 — Predictor (trial velocity) computation
 *        in the Chorin fractional-step Navier–Stokes solver.
 *
 *        This stage:
 *          - Computes u*, v*, w* using explicit integration
 *          - Applies viscosity, body forces, and gravity
 *          - Updates ONLY fluid cells (mask == 1)
 *          - Preserves boundary/wall/solid states from Pre-Step
 */

#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include <cstddef>
#include <vector>
#include "grid_math.hpp"

namespace navier_stokes_solver {

/**
 * @brief Fluid material properties used in the predictor step.
 *
 * nu      — kinematic viscosity  
 * density — mass density (ρ)
 */
struct FluidProperties {
    double nu;
    double density;
};

/**
 * @brief Computes the trial velocity field (u*, v*, w*) using explicit
 *        temporal integration of the momentum equation.
 *
 * Responsibilities:
 *   - Operate strictly on fluid cells (mask == 1)
 *   - Apply viscous diffusion (via Laplacian)
 *   - Apply external body forces fx, fy, fz
 *   - Apply gravity vector components
 *   - Preserve solid/wall states (mask == 0 or -1)
 *
 * @param dims     Grid dimensions and spacing (nx, ny, nz, dx, dy, dz)
 * @param fluid    Fluid properties (ν, ρ)
 * @param dt       Time step
 * @param u,v,w    Input velocity fields at time n
 * @param fx,fy,fz External body forces
 * @param gravity  Gravity vector [gx, gy, gz]
 * @param mask     Domain mask: 1=fluid, 0=solid, -1=wall
 * @param u_star,v_star,w_star Output trial velocities
 */
void compute_trial_velocities(
    const GridDimensions& dims,
    const FluidProperties& fluid,
    double dt,
    const double* u, const double* v, const double* w,
    const double* fx, const double* fy, const double* fz,
    const std::vector<double>& gravity,
    const std::vector<int>& mask,
    double* u_star, double* v_star, double* w_star
);

} // namespace navier_stokes_solver

#endif // PREDICTOR_HPP

