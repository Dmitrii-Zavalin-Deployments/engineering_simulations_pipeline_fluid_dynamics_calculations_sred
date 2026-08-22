/**
 * @file rhie_chow.hpp
 * @brief Declaration of the Rhie-Chow Interpolator class and GridConfig structure
 *        for collocated 3D Navier-Stokes grid stabilization.
 */

#pragma once

#include <vector>

namespace navier_stokes_solver {

class RhieChowInterpolator {
public:
    struct GridConfig {
        int nx, ny, nz;
        double dx, dy, dz;
        double dt;
    };

    /**
     * @brief Computes Rhie-Chow interpolated face velocities for a collocated 3D grid.
     * @param u Cell-centered x-velocity
     * @param v Cell-centered y-velocity
     * @param w Cell-centered z-velocity
     * @param p Cell-centered pressure
     * @param a_p Momentum diagonal coefficients (approximate cell-center inertial terms)
     * @param mask Grid cell mask for fluid-solid boundary differentiation
     * @param config Grid dimensions and time-step sizing
     * @param u_face Output x-velocities evaluated at x-faces
     * @param v_face Output y-velocities evaluated at y-faces
     * @param w_face Output z-velocities evaluated at z-faces
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
