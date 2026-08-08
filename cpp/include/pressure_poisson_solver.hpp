#ifndef PRESSURE_POISSON_SOLVER_HPP
#define PRESSURE_POISSON_SOLVER_HPP

#include <vector>
#include <string>

struct BoundaryCondition {
    std::string location; // "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall", etc.
    std::string type;     // "no-slip", "free-slip", "inflow", "outflow", "pressure"
    double scalar_p;      // Prescribed pressure value for Dirichlet boundaries
};

// Core iterative solver (interior fluid cells only)
void solve_poisson_red_black_parallel(
    std::vector<double>& p,
    const std::vector<double>& rhs,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    int max_iters, double tol
);

// Only dynamic synchronization helpers remain here
void apply_neumann_pressure(
    std::vector<double>& p,
    const std::string& location,
    int nx, int ny, int nz
);

void apply_solid_neumann_pressure_parallel(
    std::vector<double>& p, 
    const std::vector<int>& mask, 
    int nx, int ny, int nz
);

#endif // PRESSURE_POISSON_SOLVER_HPP
