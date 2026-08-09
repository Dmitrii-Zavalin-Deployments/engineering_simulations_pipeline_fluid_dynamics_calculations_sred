#ifndef SIMULATION_PRESTEP_HPP
#define SIMULATION_PRESTEP_HPP

#include <vector>
#include <string>

namespace navier_stokes_solver {

// Forward declaration of BoundaryCondition (defined in orchestrator.hpp)
struct BoundaryCondition;

/**
 * @brief Executes the Pre-Step static initialization phase.
 *        Directly maps explicit boundary conditions onto matching boundary cells.
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