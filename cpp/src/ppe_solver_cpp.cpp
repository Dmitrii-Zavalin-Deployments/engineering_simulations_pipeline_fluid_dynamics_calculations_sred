// cpp/src/ppe_solver_cpp.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace py = pybind11;

/**
 * High-Performance C++ Kernel for Pressure Poisson Equation (PPE) Solver
 * Using Successive Over-Relaxation (SOR) with Eulerian Mask Awareness 
 * (Fluid=1, Solid=0, Boundary=-1) and Fail-Fast Math Audits.
 */
double solve_ppe_sor_kernel_cpp(
    py::array_t<double> fields,
    py::array_t<int> mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho, double omega,
    int max_iter, double tol
) {
    auto r_fields = fields.mutable_unchecked<4>(); // [field_idx, x, y, z]
    auto r_mask = mask.unchecked<3>();             // [x, y, z]

    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const double inv_dx2 = 1.0 / dx2;
    const double inv_dy2 = 1.0 / dy2;
    const double inv_dz2 = 1.0 / dz2;
    const double rho_over_dt = rho / dt;

    // Field index mapping assumptions:
    // 0: u, 1: v, 2: w, 3: u_star, 4: v_star, 5: w_star, 6: P (FI.P), 7: P_NEXT (FI.P_NEXT)
    const int FIELD_P = 6;
    const int FIELD_P_NEXT = 7;
    const int FIELD_U_STAR = 3;
    const int FIELD_V_STAR = 4;
    const int FIELD_W_STAR = 5;

    double max_residual = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double current_max_delta = 0.0;

        // Domain sweep over active interior grid cells
        for (int z = 1; z < nz - 1; ++z) {
            for (int y = 1; y < ny - 1; ++y) {
                for (int x = 1; x < nx - 1; ++x) {

                    // 0. Eulerian Mask Check: Bypass internal solids (mask == 0) and boundaries (-1)
                    if (r_mask(x, y, z) <= 0) {
                        continue;
                    }

                    // 1. Geometry Setup & State Access
                    double p_c = r_fields(FIELD_P, x, y, z);
                    double p_old = r_fields(FIELD_P_NEXT, x, y, z);

                    // Helper lambda for mask-aware neighbor pressure retrieval with Neumann fallback (dp/dn = 0)
                    auto get_neighbor_p = [&](int nx_x, int ny_y, int nz_z) -> double {
                        if (nx_x < 0 || nx_x >= nx || ny_y < 0 || ny_y >= ny || nz_z < 0 || nz_z >= nz) {
                            return p_c;
                        }
                        if (r_mask(nx_x, ny_y, nz_z) <= 0) {
                            return p_c; // Neumann zero-gradient mirror
                        }
                        return r_fields(FIELD_P, nx_x, ny_y, nz_z);
                    };

                    // 2. Compute Laplacian of P for Rhie-Chow stabilization
                    double lap_p_n = (
                        (get_neighbor_p(x+1, y, z) - 2.0 * p_c + get_neighbor_p(x-1, y, z)) * inv_dx2 +
                        (get_neighbor_p(x, y+1, z) - 2.0 * p_c + get_neighbor_p(x, y-1, z)) * inv_dy2 +
                        (get_neighbor_p(x, y, z+1) - 2.0 * p_c + get_neighbor_p(x, y, z-1)) * inv_dz2
                    );
                    double rhie_chow_term = dt * lap_p_n;

                    // 3. Compute Divergence of v* (RHS Source Term Component)
                    double du_star_dx = (r_fields(FIELD_U_STAR, x+1, y, z) - r_fields(FIELD_U_STAR, x-1, y, z)) / (2.0 * dx);
                    double dv_star_dy = (r_fields(FIELD_V_STAR, x, y+1, z) - r_fields(FIELD_V_STAR, x, y-1, z)) / (2.0 * dy);
                    double dw_star_dz = (r_fields(FIELD_W_STAR, x, y, z+1) - r_fields(FIELD_W_STAR, x, y, z-1)) / (2.0 * dz);
                    double div_v_star = du_star_dx + dv_star_dy + dw_star_dz;

                    // --- RULE 7: FAIL-FAST MATH AUDIT (SOURCE) ---
                    if (!std::isfinite(div_v_star)) {
                        throw std::runtime_error("NaN detected in divergence source term.");
                    }

                    double rhs = rho_over_dt * (div_v_star - rhie_chow_term);

                    // 4. SOR Update Prep with Mask-Aware Neumann Stencils
                    double stencil_denom = 0.0;
                    double sum_neighbors = 0.0;

                    auto accumulate_neighbor = [&](int nx_x, int ny_y, int nz_z, double inv_h2) {
                        if (nx_x < 0 || nx_x >= nx || ny_y < 0 || ny_y >= ny || nz_z < 0 || nz_z >= nz) {
                            stencil_denom += inv_h2;
                            sum_neighbors += inv_h2 * p_old;
                            return;
                        }
                        int n_mask = r_mask(nx_x, ny_y, nz_z);
                        if (n_mask <= 0) {
                            stencil_denom += inv_h2;
                            sum_neighbors += inv_h2 * p_old; // Neumann mirror (dp/dn = 0)
                        } else {
                            stencil_denom += inv_h2;
                            sum_neighbors += inv_h2 * r_fields(FIELD_P_NEXT, nx_x, ny_y, nz_z);
                        }
                    };

                    accumulate_neighbor(x+1, y, z, inv_dx2);
                    accumulate_neighbor(x-1, y, z, inv_dx2);
                    accumulate_neighbor(x, y+1, z, inv_dy2);
                    accumulate_neighbor(x, y-1, z, inv_dy2);
                    accumulate_neighbor(x, y, z+1, inv_dz2);
                    accumulate_neighbor(x, y, z-1, inv_dz2);

                    stencil_denom *= 2.0;

                    if (stencil_denom <= 0.0) {
                        stencil_denom = 1.0e-12; // Guard against division by zero
                    }

                    // --- RULE 7: PRE-UPDATE AUDIT ---
                    if (!std::isfinite(p_old) || std::abs(p_old) > 1.0e15) {
                        throw std::runtime_error("Poisoned Pressure Trial detected (non-finite or extreme p_old).");
                    }

                    // 5. Calculate Trial Pressure via SOR Formula
                    double p_new = (1.0 - omega) * p_old + (omega / stencil_denom) * (sum_neighbors - rhs);

                    // --- RULE 7: POST-UPDATE AUDIT ---
                    if (!std::isfinite(p_new)) {
                        throw std::runtime_error("Non-finite pressure generated in SOR step.");
                    }

                    double delta = std::abs(p_new - p_old);
                    current_max_delta = std::max(current_max_delta, delta);

                    // 6. Direct write-back to target buffer
                    r_fields(FIELD_P_NEXT, x, y, z) = p_new;
                }
            }
        }

        max_residual = current_max_delta;
        if (max_residual < tol) {
            break;
        }
    }

    return max_residual;
}

