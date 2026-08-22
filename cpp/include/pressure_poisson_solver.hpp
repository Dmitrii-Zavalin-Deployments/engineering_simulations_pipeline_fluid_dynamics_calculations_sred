/**
 * @file pressure_poisson_solver.hpp
 * @brief Header for the Pressure Poisson Equation (PPE) solver and
 *        physically-correct boundary pressure handling.
 *
 *        This module provides:
 *          - Red–Black Gauss–Seidel PPE solver
 *          - Hydrostatic / gravity-balanced Neumann BCs
 *          - Solid-boundary Neumann extrapolation
 *          - Dirichlet face tracking (pressure / outflow)
 *
 *        All routines are mask-aware:
 *          mask == 1 → fluid
 *          mask == 0 → solid
 *          mask == -1 → wall
 */

#ifndef PRESSURE_POISSON_SOLVER_HPP
#define PRESSURE_POISSON_SOLVER_HPP

#include <vector>
#include <string>
#include "grid_math.hpp"
#include "boundary_condition.hpp"

namespace navier_stokes_solver {

/**
 * @brief Tracks domain faces governed by fixed (Dirichlet) pressure conditions.
 *
 * These faces must NOT receive Neumann updates.
 */
struct DirichletFaces {
    bool x_min = false;
    bool x_max = false;
    bool y_min = false;
    bool y_max = false;
    bool z_min = false;
    bool z_max = false;
};

/**
 * @brief Solves the Pressure Poisson Equation (PPE) using
 *        Red–Black Gauss–Seidel iteration.
 *
 * Responsibilities:
 *   - Operates strictly on interior fluid cells (mask == 1)
 *   - Applies Dirichlet pressure anchors (pressure / outflow BCs)
 *   - Applies gravity-balanced Neumann BCs on non-Dirichlet faces
 *   - Applies solid-boundary Neumann extrapolation (mask == 0)
 *
 * @param p        Pressure field (updated in-place)
 * @param rhs      Divergence-based source term
 * @param mask     Domain mask (fluid / solid / wall)
 * @param bc_list  Boundary condition list
 * @param nx,ny,nz Grid dimensions
 * @param dx,dy,dz Grid spacing
 * @param max_iters Maximum GS iterations
 * @param tol       Convergence tolerance
 * @param density   Fluid density
 * @param gravity   Gravity vector [gx, gy, gz]
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
 * @brief Applies hydrostatic Neumann pressure boundary conditions
 *        on domain faces that are NOT Dirichlet-anchored.
 *
 * This enforces:
 *      ∂p/∂n = ρ g_n
 *
 * where g_n is the gravity component normal to the face.
 */
void apply_neumann_pressure(
    std::vector<double>& p,
    const std::string& location,
    const DirichletFaces& dirichlet,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double density,
    const std::vector<double>& gravity
);

/**
 * @brief Applies Neumann pressure extrapolation inside solid regions (mask == 0).
 *
 * Solid cells inherit pressure from adjacent fluid cells.
 * This prevents stencil pollution in the PPE solver.
 */
void apply_solid_neumann_pressure_parallel(
    std::vector<double>& p,
    const std::vector<int>& mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz
);

} // namespace navier_stokes_solver

#endif // PRESSURE_POISSON_SOLVER_HPP

