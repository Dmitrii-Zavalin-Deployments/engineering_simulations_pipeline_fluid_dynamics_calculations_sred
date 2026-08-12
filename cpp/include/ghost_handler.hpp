/**
 * @file ghost_handler.hpp
 * @brief Header for ghost cell and buffer synchronization utilities.
 */

#ifndef GHOST_HANDLER_HPP
#define GHOST_HANDLER_HPP

#include "base_operator.hpp"
#include <cstddef>

namespace navier_stokes_solver {

/**
 * @brief Synchronizes trial buffers (STAR, NEXT) with baseline foundation values
 *        (u, v, w, p) directly in the memory space.
 * 
 * Compliance:
 * - Rule 7 & 9: Direct buffer alignment for stability.
 * - Fail-Fast foundation integrity checks.
 */
void sync_ghost_trial_buffers(
    const double* u, const double* v, const double* w, const double* p,
    double* u_star, double* v_star, double* w_star, double* p_next,
    size_t total_cells
);

} // namespace navier_stokes_solver

#endif // GHOST_HANDLER_HPP
