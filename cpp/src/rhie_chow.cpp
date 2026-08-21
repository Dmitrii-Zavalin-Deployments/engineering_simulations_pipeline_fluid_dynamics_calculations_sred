/**
 * @file rhie_chow.cpp
 * @brief Implementation of Rhie-Chow collocated grid velocity interpolation with 
 *        2nd-order central cell-centered pressure gradients to suppress checkerboard decoupling.
 */

#include "rhie_chow.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace navier_stokes_solver {

void RhieChowInterpolator::interpolateFaceVelocities(
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    const std::vector<double>& p,
    const std::vector<double>& a_p,
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

    // Helper lambda for 3D flat indexing
    auto get_idx = [nx, ny](int i, int j, int k) {
        return i + nx * (j + ny * k);
    };

    // --- 1. X-Face Velocities ---
    // Stored on faces between (i, j, k) and (i+1, j, k)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx - 1; ++i) {
                int idx_P = get_idx(i, j, k);
                int idx_E = get_idx(i + 1, j, k);
                int face_idx = i + (nx - 1) * (j + ny * k);

                // Linear interpolation of velocity to face
                double u_lin = 0.5 * (u[idx_P] + u[idx_E]);

                // Averaged momentum coefficient at face (approximate inverse of center coefficient)
                double ap_face = 0.5 * (a_p[idx_P] + a_p[idx_E]);
                double d_face = (ap_face > 0.0) ? (1.0 / ap_face) : 0.0;

                // Sharp pressure gradient at face
                double dp_dx_sharp = (p[idx_E] - p[idx_P]) / dx;
                
                // Correct 2nd-order central cell-centered pressure gradients at P and E
                double dp_dx_P = (i > 0 && i < nx - 1) ? (p[idx_E] - p[get_idx(i - 1, j, k)]) / (2.0 * dx) : dp_dx_sharp;
                double dp_dx_E = (i + 1 > 0 && i + 1 < nx - 1) ? (p[get_idx(i + 2, j, k)] - p[idx_P]) / (2.0 * dx) : dp_dx_sharp;
                double dp_dx_avg = 0.5 * (dp_dx_P + dp_dx_E);

                // Rhie-Chow correction formulation
                u_face[face_idx] = u_lin - d_face * (dp_dx_sharp - dp_dx_avg);
            }
        }
    }

    // --- 2. Y-Face Velocities ---
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny - 1; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx_P = get_idx(i, j, k);
                int idx_N = get_idx(i, j + 1, k);
                int face_idx = i + nx * (j + (ny - 1) * k);

                double v_lin = 0.5 * (v[idx_P] + v[idx_N]);
                double ap_face = 0.5 * (a_p[idx_P] + a_p[idx_N]);
                double d_face = (ap_face > 0.0) ? (1.0 / ap_face) : 0.0;

                double dp_dy_sharp = (p[idx_N] - p[idx_P]) / dy;
                double dp_dy_P = (j > 0 && j < ny - 1) ? (p[idx_N] - p[get_idx(i, j - 1, k)]) / (2.0 * dy) : dp_dy_sharp;
                double dp_dy_N = (j + 1 > 0 && j + 1 < ny - 1) ? (p[get_idx(i, j + 2, k)] - p[idx_P]) / (2.0 * dy) : dp_dy_sharp;
                double dp_dy_avg = 0.5 * (dp_dy_P + dp_dy_N);

                v_face[face_idx] = v_lin - d_face * (dp_dy_sharp - dp_dy_avg);
            }
        }
    }

    // --- 3. Z-Face Velocities ---
    for (int k = 0; k < nz - 1; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx_P = get_idx(i, j, k);
                int idx_T = get_idx(i, j, k + 1);
                int face_idx = i + nx * (j + ny * k);

                double w_lin = 0.5 * (w[idx_P] + w[idx_T]);
                double ap_face = 0.5 * (a_p[idx_P] + a_p[idx_T]);
                double d_face = (ap_face > 0.0) ? (1.0 / ap_face) : 0.0;

                double dp_dz_sharp = (p[idx_T] - p[idx_P]) / dz;
                double dp_dz_P = (k > 0 && k < nz - 1) ? (p[idx_T] - p[get_idx(i, j, k - 1)]) / (2.0 * dz) : dp_dz_sharp;
                double dp_dz_T = (k + 1 > 0 && k + 1 < nz - 1) ? (p[get_idx(i, j, k + 2)] - p[idx_P]) / (2.0 * dz) : dp_dz_sharp;
                double dp_dz_avg = 0.5 * (dp_dz_P + dp_dz_T);

                w_face[face_idx] = w_lin - d_face * (dp_dz_sharp - dp_dz_avg);
            }
        }
    }
}

} // namespace navier_stokes_solver
