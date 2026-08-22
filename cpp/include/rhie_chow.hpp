/**
 * @file rhie_chow.hpp
 * @brief Declaration of the Rhie–Chow Interpolator and GridConfig structure
 *        for stabilized collocated 3D Navier–Stokes velocity–pressure coupling.
 *
 *        This module provides:
 *          - Face velocity interpolation using Rhie–Chow correction
 *          - Mask‑aware pressure gradient evaluation
 *          - Prevention of checkerboard pressure decoupling
 *          - Boundary‑safe stencil handling
 */

#pragma once

#include <vector>

namespace navier_stokes_solver {

/**
 * @brief Rhie–Chow interpolator for collocated grids.
 *
 * Responsibilities:
 *   - Compute face velocities (u_face, v_face, w_face)
 *   - Apply Rhie–Chow pressure correction
 *   - Respect fluid/solid/wall masks
 *   - Prevent stencil pollution at boundaries
 */
class RhieChowInterpolator {
public:

    /**
     * @brief Grid configuration for Rhie–Chow interpolation.
     *
     * nx, ny, nz : grid resolution  
     * dx, dy, dz : grid spacing  
     * dt         : time step (used for inertial coefficient scaling)
     */
    struct GridConfig {
        int nx, ny, nz;
        double dx, dy, dz;
        double dt;
    };

    /**
     * @brief Computes Rhie–Chow interpolated face velocities.
     *
     * @param u      Cell‑centered x‑velocity
     * @param v      Cell‑centered y‑velocity
     * @param w      Cell‑centered z‑velocity
     * @param p      Cell‑centered pressure
     * @param a_p    Momentum diagonal coefficients (≈ ρ / dt)
     * @param mask   Grid mask: 1=fluid, 0=solid, -1=wall
     * @param config Grid dimensions and spacing
     * @param u_face Output x‑face velocities (size: (nx‑1)*ny*nz)
     * @param v_face Output y‑face velocities (size: nx*(ny‑1)*nz)
     * @param w_face Output z‑face velocities (size: nx*ny*(nz‑1))
     *
     * Notes:
     *   - Mask‑aware: face velocity = 0 if either adjacent cell is non‑fluid
     *   - Uses central gradients where possible, one‑sided near boundaries
     *   - Prevents checkerboard pressure oscillations
     */
    static void interpolateFaceVelocities(
        const std::vector<double>& u,
        const std::vector<double>& v,
        const std::vector<double>& w,
        const std::vector<double>& p,
        const std::vector<double>& a_p,
        const std::vector<int>& mask,
        const GridConfig& config,
        std::vector<double>& u_face,
        std::vector<double>& v_face,
        std::vector<double>& w_face
    );
};

} // namespace navier_stokes_solver

