/**
 * @file boundary_condition.hpp
 * @brief Shared structures for boundary conditions and boundary values.
 *
 *        This module defines:
 *          - BoundaryValues: optional per‑component values
 *          - BoundaryCondition: full BC descriptor used by Pre‑Step, PPE, and RC
 *
 *        Supported locations:
 *          "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"
 *
 *        Supported types:
 *          "no-slip", "free-slip", "inflow", "outflow", "pressure"
 */

#ifndef BOUNDARY_CONDITION_HPP
#define BOUNDARY_CONDITION_HPP

#include <string>

namespace navier_stokes_solver {

/**
 * @brief Optional component‑wise boundary values.
 *
 * has_* flags indicate whether the value is explicitly provided.
 * This prevents accidental overwriting during Pre‑Step or PPE.
 */
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

/**
 * @brief Full boundary condition descriptor.
 *
 * location — domain face or "wall"
 * type     — physical BC type (no-slip, free-slip, inflow, outflow, pressure)
 *
 * scalar_p — convenience pressure value for Dirichlet pressure BCs
 * u_val, v_val, w_val — convenience velocity values for inflow/outflow
 *
 * values — structured optional component values (preferred for multi-field BCs)
 */
struct BoundaryCondition {
    std::string location;   // "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"
    std::string type;       // "no-slip", "free-slip", "inflow", "outflow", "pressure"

    double scalar_p = 0.0;  // Dirichlet pressure anchor (if applicable)

    double u_val = 0.0;     // inflow/outflow convenience values
    double v_val = 0.0;
    double w_val = 0.0;

    BoundaryValues values;  // structured optional values
};

} // namespace navier_stokes_solver

#endif // BOUNDARY_CONDITION_HPP
