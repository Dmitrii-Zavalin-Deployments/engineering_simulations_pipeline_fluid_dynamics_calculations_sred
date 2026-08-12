/**
 * @file test_boundary_common.hpp
 * @brief Shared exceptions and helper utilities for boundary condition tests.
 */

#ifndef TEST_BOUNDARY_COMMON_HPP
#define TEST_BOUNDARY_COMMON_HPP

#include <stdexcept>
#include <string>
#include <vector>
#include "boundary_condition.hpp"

namespace navier_stokes_solver {

// Custom exception type for boundary over-constraint validation
class OverConstrainedBoundaryException : public std::runtime_error {
public:
    explicit OverConstrainedBoundaryException(const std::string& message)
        : std::runtime_error(message) {}
};

// Helper function to validate boundary configuration schemas for over-constraints
inline void validate_boundary_schema(const std::vector<BoundaryCondition>& bc_list) {
    for (const auto& bc : bc_list) {
        bool has_velocity = bc.values.has_u || bc.values.has_v || bc.values.has_w;
        bool has_pressure = bc.values.has_p;
        if (has_velocity && has_pressure) {
            throw OverConstrainedBoundaryException(
                "Over-constrained boundary detected on patch '" + bc.location + 
                "': Cannot enforce Dirichlet velocity and Dirichlet pressure simultaneously."
            );
        }
    }
}

} // namespace navier_stokes_solver

#endif // TEST_BOUNDARY_COMMON_HPP
