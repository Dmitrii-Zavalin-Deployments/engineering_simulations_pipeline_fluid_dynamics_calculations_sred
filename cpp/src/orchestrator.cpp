/**
 * @file orchestrator.cpp
 * @brief Implementation of the Navier-Stokes Time-Stepping Orchestrator with 3D Hydrostatic Pressure Splitting,
 *        execution tracing, CPU performance telemetry, and temporally consistent Rhie-Chow collocated grid interpolation.
 */

#include "orchestrator.hpp"
#include "simulation_prestep.hpp"
#include "predictor.hpp"
#include "pressure_poisson_solver.hpp"
#include "corrector.hpp"
#include "grid_math.hpp"
#include "rhie_chow.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

#ifdef NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
void NavierStokesOrchestrator::capture_debug_snapshot(
    const std::string& stage_name,
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    const std::vector<double>& p
) {
    OrchestratorDebugSnapshot snap;
    snap.stage_name = stage_name;
    snap.u = u;
    snap.v = v;
    snap.w = w;
    snap.p = p;
    snap.u_star = u_star_;
    snap.v_star = v_star_;
    snap.w_star = w_star_;
    snap.rhs = rhs_;

    debug_snapshots_.push_back(snap);

    // Safe stack-smash-free console output using std::ostringstream
    std::ostringstream oss;
    oss << "[DEBUG_DUMP] Snapshot captured: " << stage_name 
        << " (cells=" << total_cells_ << ", u_size=" << u.size() << ")\n";
    std::cout << oss.str();
}
#endif

NavierStokesOrchestrator::NavierStokesOrchestrator(const GridDimensions& dims, const SolverConfig& config)
    : dims_(dims),
      config_(config),
      total_cells_(static_cast<size_t>(dims.nx) * dims.ny * dims.nz),
      u_star_(total_cells_, 0.0),
      v_star_(total_cells_, 0.0),
      w_star_(total_cells_, 0.0),
      rhs_(total_cells_, 0.0) {
    if (total_cells_ == 0) {
        throw std::invalid_argument("GridDimensions result in zero total cells.");
    }
}

void NavierStokesOrchestrator::step(
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
) {
#ifdef NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
    debug_snapshots_.clear();
#endif

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: orchestrator.cpp | Operations (Cells): " << total_cells_ 
              << " | Grid: " << dims_.nx << "x" << dims_.ny << "x" << dims_.nz 
              << " | Active Threads: " << active_threads << "\n";

    // Initialize timers for profiling
    auto wall_start = std::chrono::high_resolution_clock::now();
    std::clock_t cpu_start = std::clock();

    // 1. PRE-STEP / BOUNDARY CONDITIONS
    auto t_pre = std::chrono::high_resolution_clock::now();
    execute_pre_step(u, v, w, p, mask, bc_list, dims_.nx, dims_.ny, dims_.nz);
    auto dur_pre = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_pre
    ).count();

#ifdef NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
    capture_debug_snapshot("pre_step", u, v, w, p);
#endif

    // 1.5. GHOST & BOUNDARY SYNCHRONIZATION
    auto t_sync1 = std::chrono::high_resolution_clock::now();
    sync_ghost_trial_buffers(
        u.data(), v.data(), w.data(), p.data(),
        u_star_.data(), v_star_.data(), w_star_.data(), rhs_.data(),
        total_cells_
    );
    auto dur_sync1 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_sync1
    ).count();

#ifdef NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
    capture_debug_snapshot("ghost_sync_1", u, v, w, p);
#endif

    // 2. PREDICTOR STEP
    auto t_pred = std::chrono::high_resolution_clock::now();
    FluidProperties fluid{mu / config_.density, config_.density};
    compute_trial_velocities(
        dims_, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity,
        mask,
        u_star_.data(), v_star_.data(), w_star_.data()
    );
    auto dur_pred = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_pred
    ).count();

#ifdef NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
    capture_debug_snapshot("predictor", u, v, w, p);
#endif

    // 3. RHIE-CHOW INTERPOLATION & PRESSURE POISSON STEP
    auto t_poisson = std::chrono::high_resolution_clock::now();
    
    RhieChowInterpolator::GridConfig rc_config{
        dims_.nx, dims_.ny, dims_.nz,
        dims_.dx, dims_.dy, dims_.dz,
        dt
    };
    std::vector<double> a_p(total_cells_, config_.density / dt);

    std::vector<double> u_face((dims_.nx - 1) * dims_.ny * dims_.nz, 0.0);
    std::vector<double> v_face(dims_.nx * (dims_.ny - 1) * dims_.nz, 0.0);
    std::vector<double> w_face(dims_.nx * dims_.ny * (dims_.nz - 1), 0.0);

    RhieChowInterpolator::interpolateFaceVelocities(
        u_star_, v_star_, w_star_, p, a_p, mask, rc_config, u_face, v_face, w_face
    );

    const double scale = config_.density / dt;

    #pragma omp parallel for collapse(3) schedule(static) if(total_cells_ > 1000)
    for (int k = 0; k < dims_.nz; ++k) {
        for (int j = 0; j < dims_.ny; ++j) {
            for (int i = 0; i < dims_.nx; ++i) {
                const size_t idx = static_cast<size_t>(
                    get_flat_index(i, j, k, dims_.nx, dims_.ny)
                );

                if (mask[idx] != 1) {
                    rhs_[idx] = 0.0;
                    continue;
                }

                // Interior cells: central differences
                if (i > 0 && i < dims_.nx - 1 &&
                    j > 0 && j < dims_.ny - 1 &&
                    k > 0 && k < dims_.nz - 1) {

                    const size_t idx_e_face = i + (dims_.nx - 1) * (j + dims_.ny * k);
                    const size_t idx_w_face = (i - 1) + (dims_.nx - 1) * (j + dims_.ny * k);
                    
                    const size_t idx_n_face = i + dims_.nx * (j + (dims_.ny - 1) * k);
                    const size_t idx_s_face = i + dims_.nx * ((j - 1) + (dims_.ny - 1) * k);
                    
                    const size_t idx_t_face = i + dims_.nx * (j + dims_.ny * k);
                    const size_t idx_b_face = i + dims_.nx * (j + dims_.ny * (k - 1));

                    const double dudx = (u_face[idx_e_face] - u_face[idx_w_face]) / dims_.dx;
                    const double dvdy = (v_face[idx_n_face] - v_face[idx_s_face]) / dims_.dy;
                    const double dwdz = (w_face[idx_t_face] - w_face[idx_b_face]) / dims_.dz;

                    rhs_[idx] = scale * (dudx + dvdy + dwdz);
                }
                // Boundary fluid cells: simple one-sided approximations
                else {
                    double dudx = 0.0;
                    double dvdy = 0.0;
                    double dwdz = 0.0;

                    if (i == 0) {
                        const size_t idx_e_face = i + (dims_.nx - 1) * (j + dims_.ny * k);
                        dudx = u_face[idx_e_face] / dims_.dx;
                    } else if (i == dims_.nx - 1) {
                        const size_t idx_w_face = (i - 1) + (dims_.nx - 1) * (j + dims_.ny * k);
                        dudx = -u_face[idx_w_face] / dims_.dx;
                    }

                    if (j == 0) {
                        const size_t idx_n_face = i + dims_.nx * (j + (dims_.ny - 1) * k);
                        dvdy = v_face[idx_n_face] / dims_.dy;
                    } else if (j == dims_.ny - 1) {
                        const size_t idx_s_face = i + dims_.nx * ((j - 1) + (dims_.ny - 1) * k);
                        dvdy = -v_face[idx_s_face] / dims_.dy;
                    }

                    if (k == 0) {
                        const size_t idx_t_face = i + dims_.nx * (j + dims_.ny * k);
                        dwdz = w_face[idx_t_face] / dims_.dz;
                    } else if (k == dims_.nz - 1) {
                        const size_t idx_b_face = i + dims_.nx * (j + dims_.ny * (k - 1));
                        dwdz = -w_face[idx_b_face] / dims_.dz;
                    }

                    rhs_[idx] = scale * (dudx + dvdy + dwdz);
                }
            }
        }
    }

#ifdef NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
    capture_debug_snapshot("rhs_assembly", u, v, w, p);
#endif

    solve_poisson_red_black_parallel(
        p, rhs_, mask, bc_list,
        dims_.nx, dims_.ny, dims_.nz,
        dims_.dx, dims_.dy, dims_.dz,
        static_cast<int>(config_.max_poisson_iterations),
        config_.poisson_tolerance,
        config_.density,
        gravity
    );

#ifdef NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
    capture_debug_snapshot("poisson", u, v, w, p);
#endif

    RhieChowInterpolator::interpolateFaceVelocities(
        u_star_, v_star_, w_star_, p, a_p, mask, rc_config, u_face, v_face, w_face
    );

    auto dur_poisson = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_poisson
    ).count();

    // 4. CORRECTOR STEP
    auto t_corr = std::chrono::high_resolution_clock::now();
    solve_corrector_parallel(
        u, v, w,
        u_star_, v_star_, w_star_,
        p, mask,
        dims_.nx, dims_.ny, dims_.nz,
        dims_.dx, dims_.dy, dims_.dz,
        dt, config_.density
    );
    auto dur_corr = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_corr
    ).count();

#ifdef NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
    capture_debug_snapshot("corrector", u, v, w, p);
#endif

    // 5. FINAL BUFFER SYNCHRONIZATION
    auto t_sync2 = std::chrono::high_resolution_clock::now();
    sync_ghost_trial_buffers(
        u.data(), v.data(), w.data(), p.data(),
        u_star_.data(), v_star_.data(), w_star_.data(), rhs_.data(),
        total_cells_
    );
    auto dur_sync2 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_sync2
    ).count();

#ifdef NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
    capture_debug_snapshot("ghost_sync_2", u, v, w, p);
#endif

    auto wall_end = std::chrono::high_resolution_clock::now();
    std::clock_t cpu_end = std::clock();

    double wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        wall_end - wall_start
    ).count();
    double cpu_ms = 1000.0 * static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    double cpu_efficiency =
        (wall_ms > 0 && active_threads > 0)
            ? (cpu_ms / (wall_ms * active_threads)) * 100.0
            : 0.0;

    std::cout << "[PERF_TIMELINE] Step Durations (ms) -> Pre-step: " << dur_pre 
              << " | Sync1: " << dur_sync1 
              << " | Predictor: " << dur_pred 
              << " | Poisson & Rhie-Chow: " << dur_poisson 
              << " | Corrector: " << dur_corr 
              << " | Sync2: " << dur_sync2 << "\n";

    std::cout << "[PERF_METRICS] Wall-Clock: " << wall_ms << " ms"
              << " | CPU Time: " << cpu_ms << " ms"
              << " | Threads: " << active_threads
              << " | CPU Efficiency: " << cpu_efficiency << "%\n";
}

} // namespace navier_stokes_solver
