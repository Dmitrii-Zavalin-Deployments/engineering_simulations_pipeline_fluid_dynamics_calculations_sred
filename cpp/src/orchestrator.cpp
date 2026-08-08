#include "orchestrator.hpp"
#include <stdexcept>

namespace ops {

NavierStokesOrchestrator::NavierStokesOrchestrator(const GridDimensions& dims, const SolverConfig& config)
    : dims_(dims), config_(config), total_cells_(dims.nx * dims.ny * dims.nz) {
    u_star_.resize(total_cells_, 0.0);
    v_star_.resize(total_cells_, 0.0);
    w_star_.resize(total_cells_, 0.0);
    rhs_.resize(total_cells_, 0.0);
}

void NavierStokesOrchestrator::step(
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
) {
    // 1. PRE-STEP: Apply static Dirichlet velocity/pressure conditions on walls (mask == -1) and solids (mask == 0)
    apply_prestep_boundary_conditions(dims_, mask, bc_list, u, v, w, p);

    // 2. PREDICTOR STEP: Compute trial velocities (u*, v*, w*) for active fluid cells (mask == 1)
    FluidProperties fluid{dt, mu / config_.density};
    compute_trial_velocities(
        dims_, fluid,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        mask,
        u_star_.data(), v_star_.data(), w_star_.data()
    );

    // 3. PRESSURE POISSON STEP: Compute RHS divergence and solve pressure field iteratively
    PoissonSolverConfig p_config{config_.max_poisson_iterations, config_.poisson_tolerance, config_.density};
    solve_pressure_poisson(
        dims_, p_config, dt,
        u_star_, v_star_, w_star_,
        mask, bc_list,
        p, rhs_
    );

    // 4. CORRECTOR STEP: Project trial velocity to divergence-free velocity field u^{n+1}
    apply_corrector_step(
        dims_, config_.density, dt,
        u_star_, v_star_, w_star_,
        p, mask,
        u, v, w
    );

    // 5. GHOST & TRIAL BUFFER SYNCHRONIZATION: Sync ghost/boundary memory regions across buffers
    sync_ghost_trial_buffers(
        u.data(), v.data(), w.data(), p.data(),
        u_star_.data(), v_star_.data(), w_star_.data(), rhs_.data(),
        total_cells_
    );
}

} // namespace ops
