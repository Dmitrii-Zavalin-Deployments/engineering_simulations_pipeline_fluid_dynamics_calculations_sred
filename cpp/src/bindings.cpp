#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Function declarations for core solver kernels
void compute_predictor_kernel_cpp(
    py::array_t<double> fields, py::array_t<int> mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho, double mu
);

double solve_ppe_sor_kernel_cpp(
    py::array_t<double> fields, py::array_t<int> mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho, double omega,
    int max_iter, double tol
);

void apply_corrector_kernel_cpp(
    py::array_t<double> fields, py::array_t<int> mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho
);

namespace py = pybind11;

PYBIND11_MODULE(navier_stokes_cpp, m) {
    m.doc() = "High-performance Navier-Stokes C++ extension module";
    
    m.def("compute_predictor_kernel_cpp", &compute_predictor_kernel_cpp, "C++ Optimized Predictor Kernel");
    m.def("solve_ppe_sor_kernel_cpp", &solve_ppe_sor_kernel_cpp, "C++ PPE SOR Solver Kernel with Mask Awareness");
    m.def("apply_corrector_kernel_cpp", &apply_corrector_kernel_cpp, "C++ Optimized Corrector Kernel");
}
