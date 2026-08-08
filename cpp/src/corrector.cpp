#include "corrector.hpp"
#include <omp.h>

void solve_corrector_parallel(
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    const std::vector<double>& u_star,
    const std::vector<double>& v_star,
    const std::vector<double>& w_star,
    const std::vector<double>& p,
    const std::vector<int>& mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho)
{
    double coeff = dt / rho;
    double idx_inv = 1.0 / (2.0 * dx);
    double idy_inv = 1.0 / (2.0 * dy);
    double idz_inv = 1.0 / (2.0 * dz);

    // Execute corrector step strictly on active interior fluid cells (mask == 1)[cite: 1]
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                
                int idx_cell = i + nx * (j + ny * k);
                
                // Skip solid cells (mask == 0) and wall boundaries (mask == -1)[cite: 1]
                if (mask[idx_cell] != 1) continue;

                // Compute central pressure gradients: dp/dx, dp/dy, dp/dz[cite: 5]
                double p_west  = p[(i - 1) + nx * (j + ny * k)];
                double p_east  = p[(i + 1) + nx * (j + ny * k)];
                double p_south = p[i + nx * ((j - 1) + ny * k)];
                double p_north = p[i + nx * ((j + 1) + ny * k)];
                double p_down  = p[i + nx * (j + ny * (k - 1))];
                double p_up    = p[i + nx * (j + ny * (k + 1))];

                double dp_dx = (p_east - p_west) * idx_inv;
                double dp_dy = (p_north - p_south) * idy_inv;
                double dp_dz = (p_up - p_down) * idz_inv;

                // Project trial velocity onto divergence-free subspace (u^(n+1) = u* - (dt/rho) * grad(p))[cite: 1, 3]
                u[idx_cell] = u_star[idx_cell] - coeff * dp_dx;
                v[idx_cell] = v_star[idx_cell] - coeff * dp_dy;
                w[idx_cell] = w_star[idx_cell] - coeff * dp_dz;
            }
        }
    }
}
