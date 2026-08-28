/**
 * @file rhie_chow.cpp
 * @brief Implementation of Rhie-Chow collocated grid velocity interpolation with 
 *        mask-aware cell-centered pressure gradients and zero-flux boundary conditioning.
 */

#include "rhie_chow.hpp"
#include "grid_math.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void RhieChowInterpolator::interpolateFaceVelocities(
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    const std::vector<double>& p,
    const std::vector<double>& a_p,
    const std::vector<int>& mask,
    const GridConfig& config,
    std::vector<double>& u_face,
    std::vector<double>& v_face,
    std::vector<double>& w_face
) {
    int nx = config.nx;
    int ny = config.ny;
    int nz = config.nz;
    double dx = config.dx;
    double dy = config.dy;
    double dz = config.dz;

    // Helper lambda for 3D flat indexing using repository standard get_flat_index
    auto get_idx = [nx, ny](int i, int j, int k) {
        return static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
    };

    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    if (!mask.empty() && mask.size() != total_cells) {
        throw std::invalid_argument("CONTRACT VIOLATION: Mask vector size mismatch in RhieChowInterpolator.");
    }

    // --- 1. X-Face Velocities ---
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx - 1; ++i) {
                size_t idx_P = get_idx(i, j, k);
                size_t idx_E = get_idx(i + 1, j, k);
                size_t face_idx = static_cast<size_t>(i + (nx - 1) * (j + ny * k));

                // Enforce zero normal flux across solid (mask == 0) and wall (mask == -1) boundaries
                if (!mask.empty() && (mask[idx_P] != 1 || mask[idx_E] != 1)) {
                    u_face[face_idx] = 0.0;
                    continue;
                }

                // Linear interpolation of velocity to face
                double u_lin = 0.5 * (u[idx_P] + u[idx_E]);

                // Averaged momentum coefficient at face
                double ap_face = 0.5 * (a_p[idx_P] + a_p[idx_E]);
                double d_face = (ap_face > 0.0) ? (1.0 / ap_face) : 0.0;

                // Sharp pressure gradient at face
                double dp_dx_sharp = (p[idx_E] - p[idx_P]) / dx;

                // Mask-aware 2nd-order central cell-centered pressure gradients at P and E
                double dp_dx_P = dp_dx_sharp;
                if (i > 0 && i < nx - 1) {
                    size_t idx_W = get_idx(i - 1, j, k);
                    if (mask.empty() || mask[idx_W] == 1) {
                        dp_dx_P = (p[idx_E] - p[idx_W]) / (2.0 * dx);
                    } else {
                        dp_dx_P = (p[idx_E] - p[idx_P]) / dx; // Fallback to one-sided/sharp if neighbor is non-fluid
                    }
                }

                double dp_dx_E = dp_dx_sharp;
                if ((i + 1) > 0 && (i + 1) < nx - 1 && (i + 2 < nx)) {
                    size_t idx_EE = get_idx(i + 2, j, k);
                    if (mask.empty() || mask[idx_EE] == 1) {
                        dp_dx_E = (p[idx_EE] - p[idx_P]) / (2.0 * dx);
                    } else {
                        dp_dx_E = (p[idx_E] - p[idx_P]) / dx; // Fallback
                    }
                }

                double dp_dx_avg = 0.5 * (dp_dx_P + dp_dx_E);

                // Rhie-Chow correction formulation
                u_face[face_idx] = u_lin - d_face * (dp_dx_sharp - dp_dx_avg);
            }
        }
    }

    // --- 2. Y-Face Velocities ---
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny - 1; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx_P = get_idx(i, j, k);
                size_t idx_N = get_idx(i, j + 1, k);
                size_t face_idx = static_cast<size_t>(i + nx * (j + (ny - 1) * k));

                // Enforce zero normal flux across solid/wall boundaries
                if (!mask.empty() && (mask[idx_P] != 1 || mask[idx_N] != 1)) {
                    v_face[face_idx] = 0.0;
                    continue;
                }

                double v_lin = 0.5 * (v[idx_P] + v[idx_N]);
                double ap_face = 0.5 * (a_p[idx_P] + a_p[idx_N]);
                double d_face = (ap_face > 0.0) ? (1.0 / ap_face) : 0.0;

                double dp_dy_sharp = (p[idx_N] - p[idx_P]) / dy;

                double dp_dy_P = dp_dy_sharp;
                if (j > 0 && j < ny - 1) {
                    size_t idx_S = get_idx(i, j - 1, k);
                    if (mask.empty() || mask[idx_S] == 1) {
                        dp_dy_P = (p[idx_N] - p[idx_S]) / (2.0 * dy);
                    } else {
                        dp_dy_P = (p[idx_N] - p[idx_P]) / dy;
                    }
                }

                double dp_dy_N = dp_dy_sharp;
                if ((j + 1) > 0 && (j + 1) < ny - 1 && (j + 2 < ny)) {
                    size_t idx_NN = get_idx(i, j + 2, k);
                    if (mask.empty() || mask[idx_NN] == 1) {
                        dp_dy_N = (p[idx_NN] - p[idx_P]) / (2.0 * dy);
                    } else {
                        dp_dy_N = (p[idx_N] - p[idx_P]) / dy;
                    }
                }

                double dp_dy_avg = 0.5 * (dp_dy_P + dp_dy_N);

                v_face[face_idx] = v_lin - d_face * (dp_dy_sharp - dp_dy_avg);
            }
        }
    }

    // --- 3. Z-Face Velocities ---
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz - 1; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx_P = get_idx(i, j, k);
                size_t idx_T = get_idx(i, j, k + 1);
                size_t face_idx = static_cast<size_t>(i + nx * (j + ny * k));

                // Enforce zero normal flux across solid/wall boundaries
                if (!mask.empty() && (mask[idx_P] != 1 || mask[idx_T] != 1)) {
                    w_face[face_idx] = 0.0;
                    continue;
                }

                double w_lin = 0.5 * (w[idx_P] + w[idx_T]);
                double ap_face = 0.5 * (a_p[idx_P] + a_p[idx_T]);
                double d_face = (ap_face > 0.0) ? (1.0 / ap_face) : 0.0;

                double dp_dz_sharp = (p[idx_T] - p[idx_P]) / dz;

                double dp_dz_P = dp_dz_sharp;
                if (k > 0 && k < nz - 1) {
                    size_t idx_B = get_idx(i, j, k - 1);
                    if (mask.empty() || mask[idx_B] == 1) {
                        dp_dz_P = (p[idx_T] - p[idx_B]) / (2.0 * dz);
                    } else {
                        dp_dz_P = (p[idx_T] - p[idx_P]) / dz;
                    }
                }

                double dp_dz_T = dp_dz_sharp;
                if ((k + 1) > 0 && (k + 1) < nz - 1 && (k + 2 < nz)) {
                    size_t idx_TT = get_idx(i, j, k + 2);
                    if (mask.empty() || mask[idx_TT] == 1) {
                        dp_dz_T = (p[idx_TT] - p[idx_P]) / (2.0 * dz);
                    } else {
                        dp_dz_T = (p[idx_T] - p[idx_P]) / dz;
                    }
                }

                double dp_dz_avg = 0.5 * (dp_dz_P + dp_dz_T);

                w_face[face_idx] = w_lin - d_face * (dp_dz_sharp - dp_dz_avg);
            }
        }
    }
}

} // namespace navier_stokes_solver
