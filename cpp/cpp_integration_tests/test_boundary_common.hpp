/**
 * @file test_boundary_common.hpp
 * @brief Shared exceptions and helper utilities for boundary condition tests written in a literate style.
 */

#ifndef TEST_BOUNDARY_COMMON_HPP
#define TEST_BOUNDARY_COMMON_HPP

#include <stdexcept>
#include <string>
#include <vector>
#include "boundary_condition.hpp"

namespace navier_stokes_solver {

/**
 * @class OverConstrainedBoundaryException
 * @brief Custom exception type thrown when a boundary specification violates 
 *        well-posedness constraints in fluid dynamics.
 */
class OverConstrainedBoundaryException : public std::runtime_error {
public:
    explicit OverConstrainedBoundaryException(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @brief Validates boundary configuration schemas against over-constraint conditions.
 * 
 * In well-posed Navier-Stokes formulations, a boundary patch cannot simultaneously 
 * enforce Dirichlet velocity and Dirichlet pressure conditions. 
 * 
 * Formally, for a given boundary patch i, let:
 *     HasVelocity_i = has_u_i OR has_v_i OR has_w_i
 *     HasPressure_i = has_p_i
 * 
 * The schema validation enforces the logical constraint:
 *     NOT (HasVelocity_i AND HasPressure_i)
 * 
 * If this condition is violated, an OverConstrainedBoundaryException is raised.
 */
inline void validate_boundary_schema(const std::vector<BoundaryCondition>& bc_list) {
    // We iterate through each boundary condition patch provided in the simulation schema.
    for (const auto& bc : bc_list) {
        // Determine if any velocity component (u, v, or w) is prescribed as Dirichlet.
        bool has_velocity = bc.values.has_u || bc.values.has_v || bc.values.has_w;
        
        // Determine if pressure (p) is prescribed as Dirichlet on the same patch.
        bool has_pressure = bc.values.has_p;
        
        // We check the over-constraint rule:
        //     if has_velocity AND has_pressure then throw exception
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
