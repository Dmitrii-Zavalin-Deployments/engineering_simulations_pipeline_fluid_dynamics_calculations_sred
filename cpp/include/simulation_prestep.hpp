/**
 * @file simulation_prestep.hpp
 * @brief Header for the Pre‑Step boundary condition application phase.
 *
 *        This stage applies:
 *          - Wall BCs (mask == -1)
 *          - Solid BCs (mask == 0)
 *          - Face‑specific BCs (x_min, x_max, y_min, y_max, z_min, z_max)
 *
 *        The implementation enforces:
 *          - Mask‑aware boundary handling
 *          - No stencil pollution
 *          - No unintended overwriting
 *          - Correct free‑slip / no‑slip semantics
 */

#ifndef SIMULATION_PRESTEP_HPP
#define SIMULATION_PRESTEP_HPP

#include <vector>
#include <string>
#include "boundary_condition.hpp"

namespace navier_stokes_solver {

/**
 * @brief Executes the Pre‑Step static initialization phase.
 *
 * Applies boundary conditions to velocity (u, v, w) and pressure (p)
 * according to:
 *
 *   mask == -1 → wall cell
 *   mask == 0  → solid cell
 *   mask == 1  → fluid cell
 *
 * BCs are split into:
 *   - Generic "wall" BCs
 *   - Explicit face BCs (x_min, x_max, ...)
 *
 * The implementation guarantees:
 *   - No invalid memory access
 *   - No double‑overwrite except intentional face‑priority rules
 *   - Physically correct free‑slip / no‑slip behavior
 */
void execute_pre_step(
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    std::vector<double>& p,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    int nx, int ny, int nz
);

} // namespace navier_stokes_solver

#endif // SIMULATION_PRESTEP_HPP

