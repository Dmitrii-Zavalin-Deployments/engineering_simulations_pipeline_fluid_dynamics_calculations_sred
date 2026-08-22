/**
 * @file scaling.hpp
 * @brief Header for temporal and density scaling factor utilities used across solver steps.
 *
 *        Provides:
 *          - dt / rho   scaling for Predictor & Corrector
 *          - rho / dt   scaling for PPE inertial coefficients
 *
 *        Both functions enforce:
 *          - positivity constraints
 *          - finite-number validation
 *          - fail-fast error signaling
 */

#ifndef SCALING_HPP
#define SCALING_HPP

#include "base_operator.hpp"

namespace navier_stokes_solver {

/**
 * @brief Returns the scaling factor (dt / rho) for the Predictor and Corrector Steps.
 *
 * Guards against:
 *   - non-positive density (rho <= 0)
 *   - non-positive time step (dt <= 0)
 *   - non-finite results (NaN / Inf)
 */
double get_dt_over_rho(double dt, double rho);

/**
 * @brief Returns the scaling factor (rho / dt) for the Pressure Poisson Equation.
 *
 * Guards against:
 *   - non-positive time step (dt <= 0)
 *   - non-positive density (rho <= 0)
 *   - non-finite results (NaN / Inf)
 */
double get_rho_over_dt(double dt, double rho);

} // namespace navier_stokes_solver

#endif // SCALING_HPP
