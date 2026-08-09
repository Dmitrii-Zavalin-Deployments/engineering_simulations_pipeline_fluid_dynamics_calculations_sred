/**
 * @file orchestrator.cpp
 * @brief Implementation of the Navier-Stokes Time-Stepping Orchestrator.
 */

#include "orchestrator.hpp"
#include "grid_math.hpp"
#include <stdexcept>

namespace navier_stokes_solver {

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
    execute_pre_step(u, v, w, p, mask, bc_list, dims_.nx, dims_.ny, dims_.nz);

    // 2. PREDICTOR STEP: Compute trial velocities (u*, v*, w*) for active fluid cells (mask == 1)
    FluidProperties fluid{mu / config_.density, config_.density};
    compute_trial_velocities(
        dims_, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        mask,
        u_star_.data(), v_star_.data(), w_star_.data()
    );

    // 3. PRESSURE POISSON STEP: Compute RHS divergence and solve pressure field iteratively
    // Compute divergence of trial velocity field into rhs_ (scaling by density / dt)
    const double scale = config_.density / dt;
    for (int k = 0; k < dims_.nz; ++k) {
        for (int j = 0; j < dims_.ny; ++j) {
            for (int i = 0; i < dims_.nx; ++i) {
                const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, dims_.nx, dims_.ny));
                if (mask[idx] == 1) {
                    // Central difference divergence of u* / dx, v* / dy, w* / dz
                    const size_t idx_east  = static_cast<size_t>(get_flat_index(i + 1, j, k, dims_.nx, dims_.ny));
                    const size_t idx_west  = static_cast<size_t>(get_flat_index(i - 1, j, k, dims_.nx, dims_.ny));
                    const size_t idx_north = static_cast<size_t>(get_flat_index(i, j + 1, k, dims_.nx, dims_.ny));
                    const size_t idx_south = static_cast<size_t>(get_flat_index(i, j - 1, k, dims_.nx, dims_.ny));
                    const size_t idx_up    = static_cast<size_t>(get_flat_index(i, j, k + 1, dims_.nx, dims_.ny));
                    const size_t idx_down  = static_cast<size_t>(get_flat_index(i, j, k - 1, dims_.nx, dims_.ny));

                    double dudx = (u_star_[idx_east]  - u_star_[idx_west])  / (2.0 * dims_.dx);
                    double dvdy = (v_star_[idx_north] - v_star_[idx_south]) / (2.0 * dims_.dy);
                    double dwdz = (w_star_[idx_up]    - w_star_[idx_down])  / (2.0 * dims_.dz);

                    rhs_[idx] = scale * (dudx + dvdy + dwdz);
                } else {
                    rhs_[idx] = 0.0;
                }
            }
        }
    }

    solve_poisson_red_black_parallel(
        p, rhs_, mask, bc_list,
        dims_.nx, dims_.ny, dims_.nz,
        dims_.dx, dims_.dy, dims_.dz,
        static_cast<int>(config_.max_poisson_iterations),
        config_.poisson_tolerance
    );

    // 4. CORRECTOR STEP: Project trial velocity to divergence-free velocity field u^{n+1}
    solve_corrector_parallel(
        u, v, w,
        u_star_, v_star_, w_star_,
        p, mask,
        dims_.nx, dims_.ny, dims_.nz,
        dims_.dx, dims_.dy, dims_.dz,
        dt, config_.density
    );

    // 5. GHOST & TRIAL BUFFER SYNCHRONIZATION: Sync ghost/boundary memory regions across buffers
    sync_ghost_trial_buffers(
        u.data(), v.data(), w.data(), p.data(),
        u_star_.data(), v_star_.data(), w_star_.data(), rhs_.data(),
        total_cells_
    );
}

} // namespace navier_stokes_solver
