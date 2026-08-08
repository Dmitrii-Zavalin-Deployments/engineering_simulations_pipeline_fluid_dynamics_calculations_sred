#ifndef ORCHESTRATOR_HPP
#define ORCHESTRATOR_HPP

#include <vector>
#include <string>
#include <cstddef>
#include "predictor.hpp"

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
     * 1. Pre-Step: Apply Dirichlet velocity & pressure boundary conditions (mask == -1)
     * 2. Predictor Step: Compute trial velocities u* (mask == 1)
     * 3. Poisson Solver: Parallel Red-Black Gauss-Seidel with Solid & Neumann sync
     * 4. Corrector Step: Project trial velocities to divergence-free field u^{n+1}
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
    void apply_pre_step_bcs(
        const std::vector<int>& mask,
        const std::vector<BoundaryCondition>& bc_list,
        std::vector<double>& u,
        std::vector<double>& v,
        std::vector<double>& w,
        std::vector<double>& p
    );

    void compute_divergence_rhs(
        const std::vector<double>& u_star,
        const std::vector<double>& v_star,
        const std::vector<double>& w_star,
        const std::vector<int>& mask,
        double dt,
        std::vector<double>& rhs
    );

    void solve_poisson(
        std::vector<double>& p,
        const std::vector<double>& rhs,
        const std::vector<int>& mask,
        const std::vector<BoundaryCondition>& bc_list
    );

    void apply_corrector(
        const std::vector<double>& u_star,
        const std::vector<double>& v_star,
        const std::vector<double>& w_star,
        const std::vector<double>& p,
        const std::vector<int>& mask,
        double dt,
        std::vector<double>& u,
        std::vector<double>& v,
        std::vector<double>& w
    );

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
