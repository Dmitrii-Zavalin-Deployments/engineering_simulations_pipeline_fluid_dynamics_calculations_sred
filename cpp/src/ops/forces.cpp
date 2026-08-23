/**
 * @file forces.cpp
 * @brief Implementation of body force validation and retrieval with execution tracing.
 */

#include "forces.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

std::array<double, 3> validate_and_get_forces(const std::vector<double>& forces) {
    // 1. Contract & Dimension Guard
    if (forces.size() != 3) {
        std::cerr << "CONTRACT VIOLATION: Invalid force vector length: " << forces.size() 
                  << ". Expected 3 components (Fx, Fy, Fz).\n";
        throw std::invalid_argument("Invalid body force vector size.");
    }

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: forces.cpp | Operations (Forces): 3" 
              << " | Active Threads: " << active_threads << "\n";

    // 2. Forensic Numerical Audit
    for (size_t i = 0; i < 3; ++i) {
        if (!std::isfinite(forces[i])) {
            std::cerr << "MATH FAILURE: Non-finite body force detected at index " << i 
                      << " | Value: " << forces[i] << "\n";
            throw std::runtime_error("Body force is non-finite.");
        }
    }

    return {forces[0], forces[1], forces[2]};
}

} // namespace navier_stokes_solver
