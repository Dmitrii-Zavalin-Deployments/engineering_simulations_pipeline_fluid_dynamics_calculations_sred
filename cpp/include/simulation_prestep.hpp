#ifndef SIMULATION_PRESTEP_HPP
#define SIMULATION_PRESTEP_HPP

#include <vector>
#include <string>

// Mirrors the "values" object in the JSON schema
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

// Mirrors the boundary condition schema object
struct BoundaryCondition {
    std::string location; // "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"
    std::string type;     // "no-slip", "free-slip", "inflow", "outflow", "pressure"
    BoundaryValues values;
};

/**
 * @brief Executes the Pre-Step static initialization phase.
 *        Directly maps the explicit schema 'values' (u, v, w, p) onto 
 *        matching boundary cells based on configuration data.
 */
void execute_pre_step(
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    std::vector<double>& p,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    int nx, int ny, int nz
);

#endif // SIMULATION_PRESTEP_HPP
