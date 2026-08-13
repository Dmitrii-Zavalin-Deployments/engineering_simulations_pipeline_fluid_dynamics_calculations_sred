/**
 * @file pressure_poisson_solver.hpp
 * @brief Header for iterative Pressure Poisson Equation (PPE) solvers and boundary condition applications.
 */

#ifndef PRESSURE_POISSON_SOLVER_HPP
#define PRESSURE_POISSON_SOLVER_HPP

#include <vector>
#include <string>
#include "grid_math.hpp"
#include "boundary_condition.hpp"

namespace navier_stokes_solver {

/**
 * @brief Solves the Pressure Poisson Equation (PPE) iteratively using Red-Black Gauss-Seidel parallelization.
 *        Operates strictly on interior fluid cells while respecting boundary conditions, body forces, and masks.
 */
void solve_poisson_red_black_parallel(
    std::vector<double>& p,
    const std::vector<double>& rhs,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    int max_iters, double tol,
    double density,
    const std::vector<double>& gravity
);

/**
 * @brief Applies rigorous Neumann pressure boundary conditions with hydrostatic and body-force balancing.
 */
void apply_neumann_pressure(
    std::vector<double>& p,
    const std::string& location,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double density,
    const std::vector<double>& gravity
);

/**
 * @brief Applies Neumann pressure conditions across solid boundaries in parallel.
 */
void apply_solid_neumann_pressure_parallel(
    std::vector<double>& p, 
    const std::vector<int>& mask, 
    int nx, int ny, int nz,
    double dx, double dy, double dz
);

} // namespace navier_stokes_solver

#endif // PRESSURE_POISSON_SOLVER_HPP
