// cpp/src/corrector_kernel.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

/**
 * @brief Projects the intermediate velocity field u* onto a divergence-free subspace.
 * Formula: u^{n+1} = u^* - (dt / rho) * grad(p^{n+1})
 * Bypasses internal solids and boundary ghost cells (mask != 1).
 */
void apply_corrector_kernel_cpp(
    py::array_t<double> fields,
    py::array_t<int> mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho
) {
    auto r_fields = fields.mutable_unchecked<4>(); // [field_idx, x, y, z]
    auto r_mask = mask.unchecked<3>();             // [x, y, z]

    const double scaling = dt / rho;
    const double idx = 1.0 / (2.0 * dx);
    const double idy = 1.0 / (2.0 * dy);
    const double idz = 1.0 / (2.0 * dz);

    #pragma omp parallel for collapse(3)
    for (int z = 1; z < nz - 1; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                
                // MASK FILTER: Only update active fluid cells (mask == 1)
                if (r_mask(x, y, z) != 1) continue;

                // Field Index Mapping:
                // 0=u, 1=v, 2=w, 3=u_star, 4=v_star, 5=w_star, 6=p_next (Pressure)
                double dp_dx = (r_fields(6, x + 1, y, z) - r_fields(6, x - 1, y, z)) * idx;
                double dp_dy = (r_fields(6, x, y + 1, z) - r_fields(6, x, y - 1, z)) * idy;
                double dp_dz = (r_fields(6, x, y, z + 1) - r_fields(6, x, y, z - 1)) * idz;

                double u_star = r_fields(3, x, y, z);
                double v_star = r_fields(4, x, y, z);
                double w_star = r_fields(5, x, y, z);

                // Apply velocity correction in-place
                r_fields(3, x, y, z) = u_star - scaling * dp_dx;
                r_fields(4, x, y, z) = v_star - scaling * dp_dy;
                r_fields(5, x, y, z) = w_star - scaling * dp_dz;
            }
        }
    }
}
