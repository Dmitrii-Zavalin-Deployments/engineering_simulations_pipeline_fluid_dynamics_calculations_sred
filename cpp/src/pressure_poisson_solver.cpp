#include "orchestrator.hpp"
#include "pressure_poisson_solver.hpp"
#include <cmath>
#include <omp.h>

void apply_neumann_pressure(
    std::vector<double>& p,
    const std::string& location,
    int nx, int ny, int nz) 
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                if (location == "x_min" && i == 0) {
                    p[idx] = p[1 + nx * (j + ny * k)];
                } else if (location == "x_max" && i == nx - 1) {
                    p[idx] = p[(nx - 2) + nx * (j + ny * k)];
                } else if (location == "y_min" && j == 0) {
                    p[idx] = p[i + nx * (1 + ny * k)];
                } else if (location == "y_max" && j == ny - 1) {
                    p[idx] = p[i + nx * ((ny - 2) + ny * k)];
                } else if (location == "z_min" && k == 0) {
                    p[idx] = p[i + nx * (j + ny * 1)];
                } else if (location == "z_max" && k == nz - 1) {
                    p[idx] = p[i + nx * (j + ny * (nz - 2))];
                } else if (location == "wall") {
                    if (i == 0) p[idx] = p[1 + nx * (j + ny * k)];
                    else if (i == nx - 1) p[idx] = p[(nx - 2) + nx * (j + ny * k)];
                    else if (j == 0) p[idx] = p[i + nx * (1 + ny * k)];
                    else if (j == ny - 1) p[idx] = p[i + nx * ((ny - 2) + ny * k)];
                    else if (k == 0) p[idx] = p[i + nx * (j + ny * 1)];
                    else if (k == nz - 1) p[idx] = p[i + nx * (j + ny * (nz - 2))];
                }
            }
        }
    }
}

void apply_solid_neumann_pressure_parallel(
    std::vector<double>& p, 
    const std::vector<int>& mask, 
    int nx, int ny, int nz) 
{
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                int idx = i + nx * (j + ny * k);
                if (mask[idx] != 0) continue; // Target internal solid cells only

                double p_west  = p[(i - 1) + nx * (j + ny * k)];
                double p_east  = p[(i + 1) + nx * (j + ny * k)];
                double p_south = p[i + nx * ((j - 1) + ny * k)];
                double p_north = p[i + nx * ((j + 1) + ny * k)];
                double p_down  = p[i + nx * (j + ny * (k - 1))];
                double p_up    = p[i + nx * (j + ny * (k + 1))];

                p[idx] = (p_east + p_west + p_north + p_south + p_up + p_down) / 6.0;
            }
        }
    }
}

void solve_poisson_red_black_parallel(
    std::vector<double>& p,
    const std::vector<double>& rhs,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    int max_iters, double tol) 
{
    double idx2 = 1.0 / (dx * dx);
    double idy2 = 1.0 / (dy * dy);
    double idz2 = 1.0 / (dz * dz);
    double factor = 0.5 / (idx2 + idy2 + idz2);

    for (int iter = 0; iter < max_iters; ++iter) {
        
        // --- PASS 1: Update RED Interior Fluid Cells ---
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    if ((i + j + k) % 2 != 0) continue;
                    int idx = i + nx * (j + ny * k);
                    if (mask[idx] != 1) continue;

                    double p_west  = p[(i - 1) + nx * (j + ny * k)];
                    double p_east  = p[(i + 1) + nx * (j + ny * k)];
                    double p_south = p[i + nx * ((j - 1) + ny * k)];
                    double p_north = p[i + nx * ((j + 1) + ny * k)];
                    double p_down  = p[i + nx * (j + ny * (k - 1))];
                    double p_up    = p[i + nx * (j + ny * (k + 1))];

                    p[idx] = factor * (
                        (p_east + p_west) * idx2 +
                        (p_north + p_south) * idy2 +
                        (p_up + p_down) * idz2 -
                        rhs[idx]
                    );
                }
            }
        }

        // --- PASS 2: Update BLACK Interior Fluid Cells ---
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    if ((i + j + k) % 2 == 0) continue;
                    int idx = i + nx * (j + ny * k);
                    if (mask[idx] != 1) continue;

                    double p_west  = p[(i - 1) + nx * (j + ny * k)];
                    double p_east  = p[(i + 1) + nx * (j + ny * k)];
                    double p_south = p[i + nx * ((j - 1) + ny * k)];
                    double p_north = p[i + nx * ((j + 1) + ny * k)];
                    double p_down  = p[i + nx * (j + ny * (k - 1))];
                    double p_up    = p[i + nx * (j + ny * (k + 1))];

                    p[idx] = factor * (
                        (p_east + p_west) * idx2 +
                        (p_north + p_south) * idy2 +
                        (p_up + p_down) * idz2 -
                        rhs[idx]
                    );
                }
            }
        }

        // --- PASS 3: Synchronize Boundaries & Solids Inside Iteration ---
        #pragma omp parallel for schedule(static)
        for (size_t b = 0; b < bc_list.size(); ++b) {
            const auto& bc = bc_list[b];
            if (bc.type != "pressure") {
                apply_neumann_pressure(p, bc.location, nx, ny, nz);
            }
        }

        apply_solid_neumann_pressure_parallel(p, mask, nx, ny, nz);
    }
}
