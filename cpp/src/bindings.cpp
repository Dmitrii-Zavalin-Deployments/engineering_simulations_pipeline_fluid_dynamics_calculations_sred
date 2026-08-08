#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "predictor.hpp"

namespace py = pybind11;

// Implementation of the Python binding bridge for the predictor kernel
void compute_predictor_kernel_cpp(
    py::array_t<double> fields, py::array_t<int> mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho, double mu
) {
    auto r_fields = fields.mutable_unchecked<4>();
    
    ops::GridDimensions dims{
        static_cast<size_t>(nx), 
        static_cast<size_t>(ny), 
        static_cast<size_t>(nz), 
        dx, dy, dz
    };
    
    // FluidProperties only takes Δt and kinematic viscosity (ν = μ / ρ)
    ops::FluidProperties fluid{dt, mu / rho};

    double* u_ptr = &r_fields(0, 0, 0, 0);
    double* v_ptr = &r_fields(1, 0, 0, 0);
    double* w_ptr = &r_fields(2, 0, 0, 0);

    std::vector<double> zero_forces(nx * ny * nz, 0.0);

    ops::compute_trial_velocities(
        dims, fluid,
        u_ptr, v_ptr, w_ptr,
        zero_forces.data(), zero_forces.data(), zero_forces.data(),
        u_ptr, v_ptr, w_ptr
    );
}

PYBIND11_MODULE(navier_stokes_cpp, m) {
    m.doc() = "High-performance Navier-Stokes C++ extension module";
    
    m.def("compute_predictor_kernel_cpp", &compute_predictor_kernel_cpp, "C++ Optimized Predictor Kernel");
}