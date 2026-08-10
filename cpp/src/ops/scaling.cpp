/**
 * @file scaling.cpp
 * @brief Implementation of scaling factors for Navier-Stokes solver.
 */

#include "scaling.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace navier_stokes_solver {

double get_dt_over_rho(double dt, double rho) {
    // Rule 8: Contract Violation Check (Density cannot be zero or negative)
    if (rho <= 0.0) {
        std::cerr << "PHYSICS CRASH: Invalid density (rho=" << rho << "). "
                  << "Vacuum or negative density would cause infinite acceleration.\n";
        throw std::invalid_argument("Invalid density (rho <= 0) in scaling computation.");
    }

    double scaling = dt / rho;

    // Forensic Numerical Audit (Rule 7)
    if (!std::isfinite(scaling)) {
        std::cerr << "MATH FAILURE: dt/rho exploded (dt=" << dt << ", rho=" << rho << ").\n";
        throw std::runtime_error("Scaling factor dt/rho is non-finite.");
    }

    return scaling;
}

double get_rho_over_dt(double dt, double rho) {
    // Rule 8: Contract Violation Check (Time step cannot be zero or negative)
    if (dt <= 0.0) {
        std::cerr << "TEMPORAL CRASH: Invalid time-step (dt=" << dt << "). "
                  << "Zero or negative dt would cause infinite pressure source terms.\n";
        throw std::invalid_argument("Invalid time-step (dt <= 0) in scaling computation.");
    }

    double scaling = rho / dt;

    // Forensic Numerical Audit (Rule 7)
    if (!std::isfinite(scaling)) {
        std::cerr << "MATH FAILURE: rho/dt exploded (rho=" << rho << ", dt=" << dt << ").\n";
        throw std::runtime_error("Scaling factor rho/dt is non-finite.");
    }

    return scaling;
}

} // namespace navier_stokes_solver
