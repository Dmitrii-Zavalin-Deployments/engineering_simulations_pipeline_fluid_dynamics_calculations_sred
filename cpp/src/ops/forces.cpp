/**
 * @file forces.cpp
 * @brief Implementation of body force validation and retrieval.
 */

#include "forces.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace ops {

std::array<double, 3> validate_and_get_forces(const std::vector<double>& forces) {
    // 1. Contract & Dimension Guard (Rule 8 equivalent)
    if (forces.size() != 3) {
        std::cerr << "CONTRACT VIOLATION: Invalid force vector length: " << forces.size() 
                  << ". Expected 3 components (Fx, Fy, Fz).\n";
        throw std::invalid_argument("Invalid body force vector size.");
    }

    // 2. Forensic Numerical Audit (Rule 7 equivalent)
    for (size_t i = 0; i < 3; ++i) {
        if (!std::isfinite(forces[i])) {
            std::cerr << "MATH FAILURE: Non-finite body force detected at index " << i 
                      << " | Value: " << forces[i] << "\n";
            throw std::runtime_error("Body force is non-finite.");
        }
    }

    return {forces[0], forces[1], forces[2]};
}

} // namespace ops
