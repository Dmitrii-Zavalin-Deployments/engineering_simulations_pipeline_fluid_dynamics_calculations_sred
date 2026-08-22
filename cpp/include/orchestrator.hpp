/**
 * @file orchestrator.hpp
 * @brief Header for the NavierStokesOrchestrator, managing fractional-step solver execution.
 *
 *        The orchestrator coordinates:
 *          1. Pre-Step boundary condition application
 *          2. Predictor (trial velocity) computation
 *          3. Divergence RHS assembly and PPE solve
 *          4. Corrector projection to divergence-free velocities
 *          5. Ghost/buffer synchronization
 *
 *        It owns persistent working buffers and enforces mask-aware safety.
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

/**
 * @brief Solver configuration parameters.
 *
 * max_poisson_iterations — maximum GS iterations for PPE
 * poisson_tolerance      — convergence tolerance
 * density                — fluid density (ρ)
 */
struct SolverConfig {
    size_t max_poisson_iterations;
    double poisson_tolerance;
    double density;
};

/**
 * @brief Orchestrates one full fractional-step Navier–Stokes time step.
 *
 * Responsibilities:
 *   - Maintain persistent buffers (u*, v*, w*, rhs)
 *   - Enforce mask-aware safety at each stage
 *   - Guarantee correct ordering of solver phases
 */
class NavierStokesOrchestrator {
public:
    NavierStokesOrchestrator(const GridDimensions& dims, const SolverConfig& config);

    /**
     * @brief Executes one full time step of the fractional-step Navier-Stokes solver:
     *
     *   1. Pre-Step:
     *        Apply Dirichlet velocity & pressure BCs
     *        Apply wall/solid mask-based boundary states
     *
     *   2. Predictor Step:
     *        Compute trial velocities u* using:
     *          - advection
     *          - diffusion
     *          - body forces
     *          - gravity
     *        Applied only to fluid cells (mask == 1)
     *
     *   3. Poisson Solver:
     *        Compute divergence RHS
     *        Solve PPE iteratively using Red–Black GS
     *
     *   4. Corrector Step:
     *        Project u* → u^{n+1} using pressure gradient
     *
     *   5. Ghost/Buffer Sync:
     *        Synchronize trial buffers with persistent state
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
