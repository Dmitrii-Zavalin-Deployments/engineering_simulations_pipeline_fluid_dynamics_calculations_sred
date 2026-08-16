/**
 * @file python_gate.cpp
 * @brief Pybind11 Python bindings for the 3D Navier-Stokes C++ Orchestrator.
 * Bridges the Python sovereign SolverState container directly with the C++ engine,
 * extracting all physical constraints, domain configurations, boundary conditions, and parameters.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "orchestrator.hpp"
#include "grid_math.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace navier_stokes_solver;

namespace py = pybind11;

class PythonSolverBridge {
public:
    PythonSolverBridge(py::object state) {
        if (state.is_none()) {
            throw std::invalid_argument("FATAL ERROR: state object cannot be None.");
        }

        // 1. Extract Grid Dimensions & Spatial Bounds
        int nx = state.attr("nx").cast<int>();
        int ny = state.attr("ny").cast<int>();
        int nz = state.attr("nz").cast<int>();

        double x_min = state.attr("x_min").cast<double>();
        double x_max = state.attr("x_max").cast<double>();
        double y_min = state.attr("y_min").cast<double>();
        double y_max = state.attr("y_max").cast<double>();
        double z_min = state.attr("z_min").cast<double>();
        double z_max = state.attr("z_max").cast<double>();

        double dx = (x_max - x_min) / static_cast<double>(nx);
        double dy = (y_max - y_min) / static_cast<double>(ny);
        double dz = (z_max - z_min) / static_cast<double>(nz);

        dims_ = {nx, ny, nz, dx, dy, dz};

        // 2. Extract Fluid Properties & Solver Configuration
        py::dict fluid_props = state.attr("fluid_properties");
        double density = fluid_props["density"].cast<double>();

        py::dict config = state.attr("config");
        size_t max_poisson_iters = config["max_poisson_iterations"].cast<size_t>();
        double poisson_tolerance = config["poisson_tolerance"].cast<double>();

        config_ = {max_poisson_iters, poisson_tolerance, density};

        // Allocate persistent state buffers
        size_t total_cells = static_cast<size_t>(nx) * ny * nz;
        u_.resize(total_cells, 0.0);
        v_.resize(total_cells, 0.0);
        w_.resize(total_cells, 0.0);
        p_.resize(total_cells, 0.0);

        // 3. Initialize C++ Orchestrator Core
        orchestrator_ = std::make_unique<navier_stokes_solver::NavierStokesOrchestrator>(dims_, config_);
    }

    void step(py::object state) {
        if (state.is_none()) {
            throw std::invalid_argument("FATAL ERROR: state object cannot be None during step execution.");
        }

        int nx = dims_.nx;
        int ny = dims_.ny;
        int nz = dims_.nz;
        size_t total_cells = static_cast<size_t>(nx) * ny * nz;

        #ifdef _OPENMP
        int active_threads = omp_get_max_threads();
        #else
        int active_threads = 1;
        #endif

        std::cout << "[THREAD_TRACE] File: python_gate.cpp | Operations (Cells): " << total_cells 
                  << " | Grid: " << nx << "x" << ny << "x" << nz 
                  << " | Active Threads: " << active_threads << "\n";

        // 4. Extract Tensors & Buffers using explicit C-style array binding to guarantee in-place memory mutation
        py::array_t<double, py::array::c_style> fields = state.attr("fields").cast<py::array_t<double, py::array::c_style>>();
        py::array_t<int, py::array::c_style> mask = state.attr("mask").cast<py::array_t<int, py::array::c_style>>();

        auto r_fields = fields.mutable_unchecked<4>();
        auto r_mask = mask.unchecked<3>();

        // 5. Extract Simulation Parameters & Fluid Viscosity
        double dt = state.attr("dt").cast<double>();
        
        py::dict fluid_props = state.attr("fluid_properties");
        double mu = fluid_props["viscosity"].cast<double>();

        // 6. Extract External Forces & 3D Gravity Vector Symmetrically
        py::dict ext_forces = state.attr("external_forces");
        std::vector<double> gravity = ext_forces["gravity_vector"].cast<std::vector<double>>();
        std::vector<double> force_vec = ext_forces["force_vector"].cast<std::vector<double>>();

        if (gravity.size() != 3) {
            throw std::invalid_argument("CONTRACT VIOLATION: gravity_vector must contain exactly 3 components [gx, gy, gz].");
        }
        if (force_vec.size() != 3) {
            throw std::invalid_argument("CONTRACT VIOLATION: force_vector must contain exactly 3 components [fx, fy, fz].");
        }

        // 7. Extract Physical Constraints (Velocity & Pressure clamping limits)
        py::dict phys_constraints = state.attr("physical_constraints");
        double min_v = phys_constraints["min_velocity"].cast<double>();
        double max_v = phys_constraints["max_velocity"].cast<double>();
        double min_p = phys_constraints["min_pressure"].cast<double>();
        double max_p = phys_constraints["max_pressure"].cast<double>();

        // 8. Map NumPy fields to C++ persistent vectors for Orchestrator consumption
        std::vector<int> mask_vec(total_cells);
        std::vector<double> fx_vec(total_cells, force_vec[0]);
        std::vector<double> fy_vec(total_cells, force_vec[1]);
        std::vector<double> fz_vec(total_cells, force_vec[2]);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    u_[idx] = r_fields(0, i, j, k);
                    v_[idx] = r_fields(1, i, j, k);
                    w_[idx] = r_fields(2, i, j, k);
                    p_[idx] = r_fields(3, i, j, k);
                    
                    mask_vec[idx] = r_mask(i, j, k);
                }
            }
        }

        // 9. Extract Boundary Conditions List
        py::list py_bc_list = state.attr("boundary_conditions");
        std::vector<navier_stokes_solver::BoundaryCondition> bc_list;
        for (auto item : py_bc_list) {
            bc_list.push_back(item.cast<navier_stokes_solver::BoundaryCondition>());
        }

        // 10. Execute full time-step inside the C++ Orchestrator Core (handles non-finite safety internally)
        orchestrator_->step(dt, mu, gravity, fx_vec, fy_vec, fz_vec, mask_vec, bc_list, u_, v_, w_, p_);

        // 11. Copy modified fields back into the mutable Python NumPy array in-place with clamping
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));

                    r_fields(0, i, j, k) = std::max(min_v, std::min(max_v, u_[idx]));
                    r_fields(1, i, j, k) = std::max(min_v, std::min(max_v, v_[idx]));
                    r_fields(2, i, j, k) = std::max(min_v, std::min(max_v, w_[idx]));
                    r_fields(3, i, j, k) = std::max(min_p, std::min(max_p, p_[idx]));
                }
            }
        }
    }

    void sync_fields(py::object state) {
        if (state.is_none()) {
            throw std::invalid_argument("FATAL ERROR: state object cannot be None during sync_fields execution.");
        }

        int nx = dims_.nx;
        int ny = dims_.ny;
        int nz = dims_.nz;

        py::dict phys_constraints = state.attr("physical_constraints");
        double min_v = phys_constraints["min_velocity"].cast<double>();
        double max_v = phys_constraints["max_velocity"].cast<double>();
        double min_p = phys_constraints["min_pressure"].cast<double>();
        double max_p = phys_constraints["max_pressure"].cast<double>();

        py::array_t<double, py::array::c_style> fields = state.attr("fields").cast<py::array_t<double, py::array::c_style>>();
        auto r_fields = fields.mutable_unchecked<4>();

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    r_fields(0, i, j, k) = std::max(min_v, std::min(max_v, u_[idx]));
                    r_fields(1, i, j, k) = std::max(min_v, std::min(max_v, v_[idx]));
                    r_fields(2, i, j, k) = std::max(min_v, std::min(max_v, w_[idx]));
                    r_fields(3, i, j, k) = std::max(min_p, std::min(max_p, p_[idx]));
                }
            }
        }
    }

private:
    navier_stokes_solver::GridDimensions dims_;
    navier_stokes_solver::SolverConfig config_;
    std::unique_ptr<navier_stokes_solver::NavierStokesOrchestrator> orchestrator_;
    std::vector<double> u_;
    std::vector<double> v_;
    std::vector<double> w_;
    std::vector<double> p_;
};

PYBIND11_MODULE(navier_stokes_cpp, m) {
    m.doc() = "High-performance C++ Navier-Stokes Fractional-Step Solver Module";

    py::class_<navier_stokes_solver::BoundaryCondition>(m, "BoundaryCondition")
        .def(py::init<>())
        .def_readwrite("location", &navier_stokes_solver::BoundaryCondition::location)
        .def_readwrite("type", &navier_stokes_solver::BoundaryCondition::type)
        .def_readwrite("scalar_p", &navier_stokes_solver::BoundaryCondition::scalar_p)
        .def_readwrite("u_val", &navier_stokes_solver::BoundaryCondition::u_val)
        .def_readwrite("v_val", &navier_stokes_solver::BoundaryCondition::v_val)
        .def_readwrite("w_val", &navier_stokes_solver::BoundaryCondition::w_val);

    py::class_<PythonSolverBridge>(m, "NavierStokesSolver")
        .def(py::init<py::object>(), py::arg("state"), "Initialize solver instance directly from sovereign SolverState container.")
        .def("step", &PythonSolverBridge::step, py::arg("state"), "Advance the Navier-Stokes system by one time-step using state container references.")
        .def("sync_fields", &PythonSolverBridge::sync_fields, py::arg("state"), "Synchronize persistent C++ solution fields directly back into Python state memory.");
}
