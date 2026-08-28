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

/**
 * @brief Solver configuration parameters.
 */
struct SolverConfig {
    size_t max_poisson_iterations;
    double poisson_tolerance;
    double density;
};

/**
 * @brief Snapshot of orchestrator internal buffers for unit testing.
 */
struct OrchestratorDebugSnapshot {
    std::string stage_name;

    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> w;
    std::vector<double> p;

    std::vector<double> u_star;
    std::vector<double> v_star;
    std::vector<double> w_star;

    std::vector<double> rhs;
};

/**
 * @brief Orchestrates one full fractional-step Navier–Stokes time step.
 */
class NavierStokesOrchestrator {
public:
    NavierStokesOrchestrator(const GridDimensions& dims, const SolverConfig& config);

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

    /**
     * @brief Retrieve all debug snapshots collected during step().
     */
    const std::vector<OrchestratorDebugSnapshot>& get_debug_snapshots() const {
        return debug_snapshots_;
    }

    /**
     * @brief Capture a snapshot of internal buffers at a given stage.
     */
    void capture_debug_snapshot(
        const std::string& stage_name,
        const std::vector<double>& u,
        const std::vector<double>& v,
        const std::vector<double>& w,
        const std::vector<double>& p
    );

private:
    // CRITICAL: Scalar dimensions and configuration members MUST be declared
    // before vector buffers to guarantee correct initialization order and
    // prevent AddressSanitizer heap-buffer-overflow faults.
    GridDimensions dims_;
    SolverConfig config_;
    size_t total_cells_;
    bool cold_start_;

    // Internal persistent working buffers
    std::vector<double> u_star_;
    std::vector<double> v_star_;
    std::vector<double> w_star_;
    std::vector<double> rhs_;
    std::vector<OrchestratorDebugSnapshot> debug_snapshots_;
};

} // namespace navier_stokes_solver

#endif // ORCHESTRATOR_HPP
