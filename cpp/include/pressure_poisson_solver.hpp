#ifndef PRESSURE_POISSON_SOLVER_HPP
#define PRESSURE_POISSON_SOLVER_HPP

#include <vector>
#include <string>

namespace ops {
    struct BoundaryCondition;
}

namespace ops {

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

} // namespace ops

#endif // PRESSURE_POISSON_SOLVER_HPP
