/**
 * @file orchestrator.hpp
 * @brief Header for the NavierStokesOrchestrator, managing fractional-step solver execution.
 */

#ifndef ORCHESTRATOR_HPP
#define ORCHESTRATOR_HPP

#include <vector>
#include <string>
#include <cstddef>

#include "grid_math.hpp"
#include "boundary_condition.hpp"
#include "predictor.hpp"
#include "simulation_prestep.hpp"
#include "pressure_poisson_solver.hpp"
#include "corrector.hpp"
#include "ghost_handler.hpp"

namespace navier_stokes_solver {

struct SolverConfig {
    size_t max_poisson_iterations;
    double poisson_tolerance;
    double density;
};

class NavierStokesOrchestrator {
public:
    NavierStokesOrchestrator(const GridDimensions& dims, const SolverConfig& config);
    
    /**
     * @brief Executes one full time step of the fractional-step Navier-Stokes solver:
     * 1. Pre-Step: Apply Dirichlet velocity & pressure boundary conditions (mask == -1, mask == 0)
     * 2. Predictor Step: Compute trial velocities u* (mask == 1) via advection + diffusion + body forces + gravity
     * 3. Poisson Solver: Compute RHS divergence and solve pressure Poisson equation iteratively
     * 4. Corrector Step: Project trial velocities to divergence-free velocity field u^{n+1}
     * 5. Ghost/Buffer Sync: Synchronize intermediate trial and persistent state buffers
     */
    void step(
        double dt,
        double mu,
        const std::vector<double>& gravity,
        const std::vector<double>& fx,
        const std::vector<double>& fy,
        const std::vector<double>& fz,
        const std::vector<int>& mask,
        const std::vector<BoundaryCondition>& bc_list,
        std::vector<double>& u,
        std::vector<double>& v,
        std::vector<double>& w,
        std::vector<double>& p
    );

private:
    GridDimensions dims_;
    SolverConfig config_;
    size_t total_cells_;

    // Internal persistent working buffers
    std::vector<double> u_star_;
    std::vector<double> v_star_;
    std::vector<double> w_star_;
    std::vector<double> rhs_;
};

} // namespace navier_stokes_solver

#endif // ORCHESTRATOR_HPP
