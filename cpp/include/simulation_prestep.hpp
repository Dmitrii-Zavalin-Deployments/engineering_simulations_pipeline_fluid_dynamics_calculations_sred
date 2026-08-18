/**
 * @file simulation_prestep.hpp
 * @brief Header for the pre-step initialization phase applying boundary conditions with non-overwriting exclusivity policy.
 */

#ifndef SIMULATION_PRESTEP_HPP
#define SIMULATION_PRESTEP_HPP

#include <vector>
#include <string>
#include "boundary_condition.hpp"

namespace navier_stokes_solver {

/**
 * @brief Executes the Pre-Step static initialization phase.
 *        Directly maps explicit boundary conditions onto matching boundary cells with priority non-overwriting enforcement.
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
