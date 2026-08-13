/**
 * @file corrector.hpp
 * @brief Header for the Corrector Step of Chorin's Projection Method.
 */

#ifndef CORRECTOR_HPP
#define CORRECTOR_HPP

#include <vector>

namespace navier_stokes_solver {

/**
 * @brief Executes the Corrector Step of Chorin's Projection Method in parallel.
 * Projects tentative trial velocities onto a divergence-free vector field using the updated pressure gradient.
 * 
 * @param u Output updated x-velocity field
 * @param v Output updated y-velocity field
 * @param w Output updated z-velocity field
 * @param u_star Tentative x-velocity from predictor step
 * @param v_star Tentative y-velocity from predictor step
 * @param w_star Tentative z-velocity from predictor step
 * @param p Updated pressure field from Poisson solver
 * @param mask Domain cell classification mask (1 = Fluid, 0 = Solid, -1 = Wall)
 * @param nx Grid points in X direction
 * @param ny Grid points in Y direction
 * @param nz Grid points in Z direction
 * @param dx Spatial step size in X
 * @param dy Spatial step size in Y
 * @param dz Spatial step size in Z
 * @param dt Time step size
 * @param rho Fluid density
 */
void solve_corrector_parallel(
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    const std::vector<double>& u_star,
    const std::vector<double>& v_star,
    const std::vector<double>& w_star,
    const std::vector<double>& p,
    const std::vector<int>& mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho
);

} // namespace navier_stokes_solver

#endif // CORRECTOR_HPP
