/**
 * @file corrector.hpp
 * @brief Header for the Corrector Step of Chorin's Projection Method.
 *
 *        This stage:
 *          - Applies pressure‑gradient correction
 *          - Projects trial velocities (u*, v*, w*) onto a divergence‑free field
 *          - Uses mask‑aware gradients to avoid stencil pollution
 *          - Supports one‑sided differences at fluid–solid/wall interfaces
 */

#ifndef CORRECTOR_HPP
#define CORRECTOR_HPP

#include <vector>

namespace navier_stokes_solver {

/**
 * @brief Executes the Corrector Step of Chorin's Projection Method in parallel.
 *
 * Responsibilities:
 *   - Operate strictly on fluid cells (mask == 1)
 *   - Apply central pressure gradients in interior
 *   - Apply one‑sided gradients near solid/wall boundaries
 *   - Prevent invalid stencil access
 *   - Produce divergence‑free velocity field
 *
 * @param u       Output corrected x‑velocity
 * @param v       Output corrected y‑velocity
 * @param w       Output corrected z‑velocity
 * @param u_star  Predictor x‑velocity
 * @param v_star  Predictor y‑velocity
 * @param w_star  Predictor z‑velocity
 * @param p       Updated pressure field
 * @param mask    Domain mask: 1=fluid, 0=solid, -1=wall
 * @param nx,ny,nz Grid resolution
 * @param dx,dy,dz Grid spacing
 * @param dt      Time step
 * @param rho     Fluid density
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

