#ifndef ORCHESTRATOR_HPP
#define ORCHESTRATOR_HPP

#include <vector>
#include <string>
#include <cstddef>

#include "predictor.hpp"
#include "simulation_prestep.hpp"
#include "pressure_poisson_solver.hpp"
#include "corrector.hpp"
#include "ghost_handler.hpp"

namespace ops {

struct BoundaryCondition {
    std::string location; // "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"
    std::string type;     // "no-slip", "free-slip", "inflow", "outflow", "pressure"
    double scalar_p = 0.0;
    double u_val = 0.0;
    double v_val = 0.0;
    double w_val = 0.0;
};

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
     * 2. Predictor Step: Compute trial velocities u* (mask == 1) via advection + diffusion + body forces
     * 3. Poisson Solver: Compute RHS divergence and solve pressure Poisson equation iteratively
     * 4. Corrector Step: Project trial velocities to divergence-free velocity field u^{n+1}
     * 5. Ghost/Buffer Sync: Synchronize intermediate trial and persistent state buffers
     */
    void step(
        double dt,
        double mu,
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

} // namespace ops

#endif // ORCHESTRATOR_HPP
