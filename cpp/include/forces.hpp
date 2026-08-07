#ifndef FORCES_HPP
#define FORCES_HPP

#include "base_operator.hpp"
#include <vector>
#include <array>

/**
 * Validates and retrieves the 3-component body force vector (Fx, Fy, Fz).
 * Ensures safety, size compliance, and finiteness.
 */
std::array<double, 3> validate_and_get_forces(const std::vector<double>& forces);

#endif // FORCES_HPP
