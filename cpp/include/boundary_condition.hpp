/**
 * @file boundary_condition.hpp
 * @brief Shared structures for boundary conditions and boundary values.
 */

#ifndef BOUNDARY_CONDITION_HPP
#define BOUNDARY_CONDITION_HPP

#include <string>

namespace navier_stokes_solver {

struct BoundaryValues {
    bool has_u = false;
    double u = 0.0;
    bool has_v = false;
    double v = 0.0;
    bool has_w = false;
    double w = 0.0;
    bool has_p = false;
    double p = 0.0;
};

struct BoundaryCondition {
    std::string location; // "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"
    std::string type;     // "no-slip", "free-slip", "inflow", "outflow", "pressure"
    double scalar_p = 0.0;
    double u_val = 0.0;
    double v_val = 0.0;
    double w_val = 0.0;
    BoundaryValues values;
};

} // namespace navier_stokes_solver

#endif // BOUNDARY_CONDITION_HPP
