/**
 * @file ghost_handler.cpp
 * @brief Implementation of ghost trial buffer synchronization with execution tracing and consistent OpenMP cell threshold.
 */

#include "ghost_handler.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void sync_ghost_trial_buffers(
    const double* u, const double* v, const double* w, const double* p,
    double* u_star, double* v_star, double* w_star, double* p_next,
    size_t total_cells
) {
    // Foundation Integrity Guard (Rule 5 & 9 equivalent)
    if (!u || !v || !w || !p || !u_star || !v_star || !w_star || !p_next) {
        std::cerr << "CONTRACT VIOLATION: Cell foundation pointers are null in ghost trial buffer sync.\n";
        throw std::invalid_argument("Ghost sync failed: Cell foundation is malformed.");
    }

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: ghost_handler.cpp | Operations (Cells): " << total_cells 
              << " | Active Threads: " << active_threads << "\n";

    // Direct buffer alignment across memory space without heap reallocations
    // Added cell-count threshold guard for consistency with operator files
    #pragma omp parallel for schedule(static) if(total_cells > 1000)
    for (int64_t idx = 0; idx < static_cast<int64_t>(total_cells); ++idx) {
        u_star[idx] = u[idx];
        v_star[idx] = v[idx];
        w_star[idx] = w[idx];
        p_next[idx] = p[idx];
    }
}

} // namespace navier_stokes_solver
