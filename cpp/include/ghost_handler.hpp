/**
 * @file ghost_handler.hpp
 * @brief Header for ghost cell and buffer synchronization utilities.
 *
 *        Responsibilities:
 *          - Synchronize trial buffers (u*, v*, w*, p_next)
 *            with baseline fields (u, v, w, p)
 *          - Enforce fail‑fast integrity checks
 *          - Guarantee direct memory‑aligned copying
 *
 *        This module does NOT apply physics or masks.
 *        It only ensures buffer consistency before predictor/corrector steps.
 */

#ifndef GHOST_HANDLER_HPP
#define GHOST_HANDLER_HPP

#include "base_operator.hpp"
#include <cstddef>

namespace navier_stokes_solver {

/**
 * @brief Synchronizes trial buffers (STAR, NEXT) with baseline foundation values.
 *
 * @param u, v, w, p       Baseline fields
 * @param u_star, v_star,
 *        w_star, p_next   Output synchronized buffers
 * @param total_cells      Number of grid cells
 *
 * Notes:
 *   - Performs strict null‑pointer validation
 *   - Uses parallel copying in implementation
 *   - No mask or boundary logic is applied here
 */
void sync_ghost_trial_buffers(
    const double* u,
    const double* v,
    const double* w,
    const double* p,
    double* u_star,
    double* v_star,
    double* w_star,
    double* p_next,
    size_t total_cells
);

} // namespace navier_stokes_solver

#endif // GHOST_HANDLER_HPP

