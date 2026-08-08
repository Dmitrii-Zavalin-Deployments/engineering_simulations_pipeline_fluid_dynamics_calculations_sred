#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include "orchestrator.hpp"

namespace py = pybind11;

class PythonSolverBridge {
public:
    PythonSolverBridge(
        int nx, int ny, int nz, 
        double dx, double dy, double dz, 
        size_t max_poisson_iters, double poisson_tolerance, double density
    ) {
        dims_ = {
            nx, 
            ny, 
            nz, 
            dx, dy, dz
        };
        config_ = {max_poisson_iters, poisson_tolerance, density};
        orchestrator_ = std::make_unique<ops::NavierStokesOrchestrator>(dims_, config_);
    }

    void step(
        py::array_t<double> fields, // shape (4, nx, ny, nz) -> u, v, w, p
        py::array_t<int> mask,      // shape (nx, ny, nz)
        py::array_t<double> fx,     // shape (nx, ny, nz)
        py::array_t<double> fy,     // shape (nx, ny, nz)
        py::array_t<double> fz,     // shape (nx, ny, nz)
        const std::vector<ops::BoundaryCondition>& bc_list,
        double dt, double mu
    ) {
        auto r_fields = fields.mutable_unchecked<4>();
        auto r_mask = mask.unchecked<3>();
        auto r_fx = fx.unchecked<3>();
        auto r_fy = fy.unchecked<3>();
        auto r_fz = fz.unchecked<3>();

        size_t nx = dims_.nx;
        size_t ny = dims_.ny;
        size_t nz = dims_.nz;
        size_t total_cells = nx * ny * nz;

        // Flatten NumPy arrays to std::vector for seamless Orchestrator integration
        std::vector<double> u(total_cells);
        std::vector<double> v(total_cells);
        std::vector<double> w(total_cells);
        std::vector<double> p(total_cells);
        std::vector<int> mask_vec(total_cells);
        std::vector<double> fx_vec(total_cells);
        std::vector<double> fy_vec(total_cells);
        std::vector<double> fz_vec(total_cells);

        size_t ny_nz = ny * nz;
        size_t nz_val = nz;

        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t k = 0; k < nz; ++k) {
                    size_t idx = i * ny_nz + j * nz_val + k;
                    u[idx] = r_fields(0, i, j, k);
                    v[idx] = r_fields(1, i, j, k);
                    w[idx] = r_fields(2, i, j, k);
                    p[idx] = r_fields(3, i, j, k);
                    
                    mask_vec[idx] = r_mask(i, j, k);
                    fx_vec[idx] = r_fx(i, j, k);
                    fy_vec[idx] = r_fy(i, j, k);
                    fz_vec[idx] = r_fz(i, j, k);
                }
            }
        }

        // Execute full time-step inside the C++ Orchestrator
        orchestrator_->step(dt, mu, fx_vec, fy_vec, fz_vec, mask_vec, bc_list, u, v, w, p);

        // Copy modified fields back into the mutable Python NumPy array in-place
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t k = 0; k < nz; ++k) {
                    size_t idx = i * ny_nz + j * nz_val + k;
                    r_fields(0, i, j, k) = u[idx];
                    r_fields(1, i, j, k) = v[idx];
                    r_fields(2, i, j, k) = w[idx];
                    r_fields(3, i, j, k) = p[idx];
                }
            }
        }
    }

private:
    ops::GridDimensions dims_;
    ops::SolverConfig config_;
    std::unique_ptr<ops::NavierStokesOrchestrator> orchestrator_;
};

PYBIND11_MODULE(navier_stokes_cpp, m) {
    m.doc() = "High-performance C++ Navier-Stokes Fractional-Step Solver Module";

    py::class_<ops::BoundaryCondition>(m, "BoundaryCondition")
        .def(py::init<>())
        .def_readwrite("location", &ops::BoundaryCondition::location)
        .def_readwrite("type", &ops::BoundaryCondition::type)
        .def_readwrite("scalar_p", &ops::BoundaryCondition::scalar_p)
        .def_readwrite("u_val", &ops::BoundaryCondition::u_val)
        .def_readwrite("v_val", &ops::BoundaryCondition::v_val)
        .def_readwrite("w_val", &ops::BoundaryCondition::w_val);

    py::class_<PythonSolverBridge>(m, "NavierStokesSolver")
        .def(py::init<int, int, int, double, double, double, size_t, double, double>(), py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("dx"), py::arg("dy"), py::arg("dz"), py::arg("max_poisson_iters"), py::arg("poisson_tolerance"), py::arg("density"), "Initialize solver grid dimensions, iteration limits, and fluid density.")
        .def("step", &PythonSolverBridge::step, py::arg("fields"), py::arg("mask"), py::arg("fx"), py::arg("fy"), py::arg("fz"), py::arg("bc_list"), py::arg("dt"), py::arg("mu"), "Advance the Navier-Stokes system by one time-step (dt).");
}
