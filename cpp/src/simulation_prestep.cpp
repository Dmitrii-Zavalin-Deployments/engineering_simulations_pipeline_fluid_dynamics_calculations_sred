/**
 * @file simulation_prestep.cpp
 * @brief Implementation of Pre-Step Boundary & Initial Condition Setup.
 */

#include "orchestrator.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

inline bool matches_location(int i, int j, int k, int nx, int ny, int nz, const std::string& location) {
    if (location == "x_min" && i == 0) return true;
    if (location == "x_max" && i == nx - 1) return true;
    if (location == "y_min" && j == 0) return true;
    if (location == "y_max" && j == ny - 1) return true;
    if (location == "z_min" && k == 0) return true;
    if (location == "z_max" && k == nz - 1) return true;
    if (location == "wall") {
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
            return true;
        }
    }
    return false;
}

void execute_pre_step(
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    std::vector<double>& p,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    int nx, int ny, int nz) 
{
    // Iterate through all boundary configurations defined in the input schema array
    for (size_t b = 0; b < bc_list.size(); ++b) {
        const auto& bc = bc_list[b];
        const auto& val = bc.values;

        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));

                    // Check if current cell matches the target boundary location
                    if (!matches_location(i, j, k, nx, ny, nz, bc.location)) {
                        continue;
                    }

                    // Purely data-driven schema mapping: 
                    // Directly apply whatever component values are provided in the configuration file,
                    // regardless of whether the boundary is inflow, outflow, no-slip, or free-slip.
                    if (val.has_u) u[idx] = val.u;
                    if (val.has_v) v[idx] = val.v;
                    if (val.has_w) w[idx] = val.w;
                    if (val.has_p) p[idx] = val.p;
                }
            }
        }
    }
}

} // namespace navier_stokes_solver
