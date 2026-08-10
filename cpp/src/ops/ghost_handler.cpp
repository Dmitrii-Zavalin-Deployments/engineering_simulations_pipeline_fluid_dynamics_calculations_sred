/**
 * @file ghost_handler.cpp
 * @brief Implementation of ghost trial buffer synchronization.
 */

#include "ghost_handler.hpp"
#include <iostream>
#include <stdexcept>

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
        throw std::runtime_error("Ghost sync failed: Cell foundation is malformed.");
    }

    // Direct buffer alignment across memory space without heap reallocations
    #pragma omp parallel for
    for (size_t idx = 0; idx < total_cells; ++idx) {
        u_star[idx] = u[idx];
        v_star[idx] = v[idx];
        w_star[idx] = w[idx];
        p_next[idx] = p[idx];
    }
}

} // namespace navier_stokes_solver
