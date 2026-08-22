/**
 * @file forces.hpp
 * @brief Header for body force validation and retrieval utilities.
 *
 *        This module ensures:
 *          - Correct 3‑component force vector sizing
 *          - Finite‑number validation (no NaN / Inf)
 *          - Safe extraction for use in predictor step
 */

#ifndef FORCES_HPP
#define FORCES_HPP

#include "base_operator.hpp"
#include <vector>
#include <array>

namespace navier_stokes_solver {

/**
 * @brief Validates and retrieves the 3‑component body force vector (Fx, Fy, Fz).
 *
 * @param forces Input vector expected to contain exactly 3 components.
 * @return std::array<double,3> Validated force vector.
 *
 * Validation guarantees:
 *   - Size == 3
 *   - All components are finite
 *   - No implicit defaults or silent corrections
 *
 * Throws std::invalid_argument on any violation.
 */
std::array<double, 3> validate_and_get_forces(const std::vector<double>& forces);

} // namespace navier_stokes_solver

#endif // FORCES_HPP