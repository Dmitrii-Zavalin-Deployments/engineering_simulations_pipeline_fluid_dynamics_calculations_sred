// cpp/src/predictor_kernel.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

/**
 * High-performance C++ Predictor Kernel mirroring src/step3/predictor.py.
 * Computes intermediate trial velocity v* = v^n + (dt/rho) * (mu * lap(v^n) - rho * (v^n ⋅ ∇)v^n + F)
 * 
 * Compliance:
 * - Eulerian Mask Filtering: Bypasses solids (mask == 0) and boundary ghost cells (mask == -1).
 * - Fail-Fast Validation: Throws runtime error on non-finite numerical instabilities.
 * - In-place update via contiguous NumPy memory mapping.
 */
void compute_predictor_kernel_cpp(
    py::array_t<double> fields,
    py::array_t<int> mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho, double mu
) {
    auto r_fields = fields.mutable_unchecked<4>(); // [field_idx, x, y, z]
    auto r_mask = mask.unchecked<3>();             // [x, y, z]

    const double idx2 = 1.0 / (dx * dx);
    const double idy2 = 1.0 / (dy * dy);
    const double idz2 = 1.0 / (dz * dz);
    
    const double idx = 1.0 / (2.0 * dx);
    const double idy = 1.0 / (2.0 * dy);
    const double idz = 1.0 / (2.0 * dz);

    const double dt_over_rho = dt / rho;

    #pragma omp parallel for collapse(3)
    for (int z = 1; z < nz - 1; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                
                // --- EULERIAN MASK FILTER ---
                // Skip solids (mask == 0) and domain boundaries/ghost cells (mask == -1)
                if (r_mask(x, y, z) <= 0) {
                    continue;
                }

                // 1. Retrieve current velocity components (VX=0, VY=1, VZ=2)
                double u = r_fields(0, x, y, z);
                double v = r_fields(1, x, y, z);
                double w = r_fields(2, x, y, z);

                // 2. Compute Laplacian diffusion operator (lap_v)
                double lap_u = (r_fields(0, x+1, y, z) - 2.0*u + r_fields(0, x-1, y, z)) * idx2 +
                               (r_fields(0, x, y+1, z) - 2.0*u + r_fields(0, x, y-1, z)) * idy2 +
                               (r_fields(0, x, y, z+1) - 2.0*u + r_fields(0, x, y, z-1)) * idz2;

                double lap_v = (r_fields(1, x+1, y, z) - 2.0*v + r_fields(1, x-1, y, z)) * idx2 +
                               (r_fields(1, x, y+1, z) - 2.0*v + r_fields(1, x, y-1, z)) * idy2 +
                               (r_fields(1, x, y, z+1) - 2.0*v + r_fields(1, x, y, z-1)) * idz2;

                double lap_w = (r_fields(2, x+1, y, z) - 2.0*w + r_fields(2, x-1, y, z)) * idx2 +
                               (r_fields(2, x, y+1, z) - 2.0*w + r_fields(2, x, y-1, z)) * idy2 +
                               (r_fields(2, x, y, z+1) - 2.0*w + r_fields(2, x, y, z-1)) * idz2;

                // 3. Compute Advection operator (v^n ⋅ ∇)v^n via central differences
                double u_x = (r_fields(0, x+1, y, z) - r_fields(0, x-1, y, z)) * idx;
                double u_y = (r_fields(0, x, y+1, z) - r_fields(0, x, y-1, z)) * idy;
                double u_z = (r_fields(0, x, y, z+1) - r_fields(0, x, y, z-1)) * idz;
                double adv_u = u * u_x + v * u_y + w * u_z;

                double v_x = (r_fields(1, x+1, y, z) - r_fields(1, x-1, y, z)) * idx;
                double v_y = (r_fields(1, x, y+1, z) - r_fields(1, x, y-1, z)) * idy;
                double v_z = (r_fields(1, x, y, z+1) - r_fields(1, x, y, z-1)) * idz;
                double adv_v = u * v_x + v * v_y + w * v_z;

                double w_x = (r_fields(2, x+1, y, z) - r_fields(2, x-1, y, z)) * idx;
                double w_y = (r_fields(2, x, y+1, z) - r_fields(2, x, y-1, z)) * idy;
                double w_z = (r_fields(2, x, y, z+1) - r_fields(2, x, y, z-1)) * idz;
                double adv_w = u * w_x + v * w_y + w * w_z;

                // Body forces (expand if external force vector field is mapped)
                double fx = 0.0;
                double fy = 0.0;
                double fz = 0.0;

                // 4. Compute trial velocity components (v*)
                double u_star = u + dt_over_rho * (mu * lap_u - rho * adv_u + fx);
                double v_star = v + dt_over_rho * (mu * lap_v - rho * adv_v + fy);
                double w_star = w + dt_over_rho * (mu * lap_w - rho * adv_w + fz);

                // --- FAIL-FAST MATH CHECK ---
                if (!std::isfinite(u_star) || !std::isfinite(v_star) || !std::isfinite(w_star)) {
                    throw std::runtime_error("PREDICTOR FAILURE: Non-finite v_star detected in C++ kernel.");
                }

                // 5. Commit to Trial (Star) buffers (VX_STAR=3, VY_STAR=4, VZ_STAR=5)
                r_fields(3, x, y, z) = u_star;
                r_fields(4, x, y, z) = v_star;
                r_fields(5, x, y, z) = w_star;
            }
        }
    }
}

