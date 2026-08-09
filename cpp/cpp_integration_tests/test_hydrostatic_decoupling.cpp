/**
 * @file test_hydrostatic_decoupling.cpp
 * @brief Literate Integration Test Suite verifying Hydrostatic Pressure Splitting,
 *        Body Force Balance, and Floating-Point Stability in Static Fluid Columns.
 * 
 * LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
 * ---------------------------------------------------------------------------------
 * When modeling incompressible fluid dynamics under significant gravitational 
 * fields (such as a deep fluid column with height $H = 100\text{ m}$ and gravity 
 * $\mathbf{g} = (0, -9.81, 0)\text{ m/s}^2$), total pressure gradients can span 
 * millions of Pascals. If handled naively, numerical truncation errors in evaluating 
 * $-\frac{1}{\rho}\nabla p$ can generate artificial velocities (spurious currents) 
 * that overwhelm subtle dynamic fluctuations.
 * 
 * To eliminate these artifacts, the orchestrator implements **Hydrostatic Pressure Splitting**:
 *     (1) Total Pressure Decomposition:
 *         p_{\text{total}} = p_{\text{hydro}} + p_{\text{dynamic}}
 * 
 *     (2) Hydrostatic Balance Enforcement:
 *         \nabla p_{\text{hydro}} = \rho \mathbf{g} \implies -\frac{1}{\rho}\nabla p_{\text{hydro}} + \mathbf{g} = 0
 * 
 *     (3) Governing Momentum Balance for Quiescent Equilibrium:
 *         rho * \frac{\partial \mathbf{u}}{\partial t} = -\nabla p_{\text{dynamic}} + \mu \nabla^2 \mathbf{u} + \rho \mathbf{g} - \nabla p_{\text{hydro}} = 0
 * 
 * EXTENDED HYDROSTATIC STABILITY VALIDATION OBJECTIVE:
 * This integration test initializes a completely quiescent fluid column ($\mathbf{u} = 0$), 
 * executes $N = 100$ consecutive time steps through the orchestrator, and verifies three pillars:
 *     1. Force Balance Residual: Exact cancellation between body forces and static pressure gradients.
 *     2. Dynamic Pressure Boundedness: The dynamic pressure $p_{\text{dynamic}}$ remains strictly zero.
 *     3. Zero Spurious Currents: Velocity field infinity norm stays below machine precision ($\Vert{}\mathbf{u}^n\Vert{}_{\infty} < 10^{-14}\text{ m/s}$).
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include "orchestrator.hpp"

using json = nlohmann::json;
using namespace ops;

class HydrostaticDecouplingTest : public ::testing::Test {
protected:
    // Loads domain parameters, physical properties, and execution tolerances 
    // from external JSON configuration files using explicit repository paths.
    void SetUp() override {
        std::ifstream config_stream("cpp/cpp_integration_tests/data/config.json");
        ASSERT_TRUE(config_stream.is_open()) << "Failed to open cpp/cpp_integration_tests/data/config.json";
        config_stream >> config_json_;

        std::ifstream input_stream("cpp/cpp_integration_tests/data/navier_stokes_input.json");
        ASSERT_TRUE(input_stream.is_open()) << "Failed to open cpp/cpp_integration_tests/data/navier_stokes_input.json";
        input_stream >> input_json_;
    }

    json config_json_;
    json input_json_;
};

// =================================================================================
// Scenario 2.1: Quiescent Fluid in Deep Gravity Well ($\mathbf{f} = \mathbf{g}$)
// =================================================================================
TEST_F(HydrostaticDecouplingTest, QuiescentFluidDeepGravityWell) {
    // -----------------------------------------------------------------------------
    // Step 1: Initialize grid dimensions, physical properties, and deep well parameters.
    // Spatial grid increments are derived as:
    //     dx = (x_max - x_min) / nx
    //     dy = (y_max - y_min) / ny
    //     dz = (z_max - z_min) / nz
    // -----------------------------------------------------------------------------
    int nx = input_json_["grid"]["nx"];
    int ny = input_json_["grid"]["ny"];
    int nz = input_json_["grid"]["nz"];
    double x_min = input_json_["grid"]["x_min"];
    double x_max = input_json_["grid"]["x_max"];
    double y_min = input_json_["grid"]["y_min"];
    double y_max = input_json_["grid"]["y_max"];
    double z_min = input_json_["grid"]["z_min"];
    double z_max = input_json_["grid"]["z_max"];

    double dx = (x_max - x_min) / nx;
    double dy = (y_max - y_min) / ny;
    double dz = (z_max - z_min) / nz;

    GridDimensions dims{nx, ny, nz, dx, dy, dz};

    // Configure deep gravity well parameters explicitly for this test scenario:
    //   - Column height: H = 100.0 m (overriding domain y bounds or scaling gravity)
    //   - Fluid density: rho = 1000.0 kg/m^3
    //   - Gravity vector: g = (0.0, -9.81, 0.0) m/s^2
    double density = 1000.0;
    double mu = input_json_["fluid_properties"]["viscosity"];
    double dt = 0.01; // Controlled integration step size

    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = config_json_["max_poisson_iterations"];
    config.poisson_tolerance = config_json_["poisson_tolerance"];

    NavierStokesOrchestrator orchestrator(dims, config);

    // -----------------------------------------------------------------------------
    // Step 2: Allocate vector fields and initialize with quiescent conditions:
    //     u(x, y, z) = 0.0
    //     v(x, y, z) = 0.0
    //     w(x, y, z) = 0.0
    //     p_dynamic(x, y, z) = 0.0
    // -----------------------------------------------------------------------------
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p_dynamic(total_cells, 0.0);

    // External force vector configured with uniform downward gravity g = (0, -9.81, 0)
    double gravity_y = -9.81;
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, density * gravity_y); // f_y = rho * g_y
    std::vector<double> fz(total_cells, 0.0);

    std::vector<int> mask;
    mask.reserve(total_cells);
    for (const auto& val : input_json_["mask"]) {
        mask.push_back(val.get<int>());
    }

    std::vector<BoundaryCondition> bc_list;
    for (const auto& bc_item : input_json_["boundary_conditions"]) {
        BoundaryCondition bc;
        bc.location = bc_item["location"].get<std::string>();
        bc.type = bc_item["type"].get<std::string>();
        bc.u_val = 0.0;
        bc.v_val = 0.0;
        bc.w_val = 0.0;
        bc_list.push_back(bc);
    }

    // -----------------------------------------------------------------------------
    // Step 3: Execute Extended Time-Stepping Loop Under Hydrostatic Equilibrium (N = 100).
    // -----------------------------------------------------------------------------
    const int total_time_steps = 100;

    for (int step = 0; step < total_time_steps; ++step) {
        orchestrator.step(dt, mu, fx, fy, fz, mask, bc_list, u, v, w, p_dynamic);
    }

    // -----------------------------------------------------------------------------
    // Step 4: Assertions & Physical Validation
    // -----------------------------------------------------------------------------
    
    // Assertion 1: Body force acceleration and static pressure gradient cancel completely.
    // Specifically, we check that: || - (1 / rho) * grad(p_hydro) + g || < 1e-12
    double max_force_residual = orchestrator.ComputeMaxForceResidual();
    EXPECT_LT(max_force_residual, 1e-12) 
        << "Assertion 1 Failed: Hydrostatic pressure gradient and body force did not balance.";

    // Assertion 2: Dynamic pressure field remains zero (p_dynamic = 0) within machine precision.
    double max_dynamic_pressure = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        ASSERT_FALSE(std::isnan(p_dynamic[idx]) || std::isinf(p_dynamic[idx]))
            << "Numerical Instability: Dynamic pressure became NaN or Inf.";
        max_dynamic_pressure = std::max(max_dynamic_pressure, std::abs(p_dynamic[idx]));
    }
    EXPECT_LT(max_dynamic_pressure, 1e-12) 
        << "Assertion 2 Failed: Non-zero dynamic pressure developed in a quiescent fluid column.";

    // Assertion 3: Velocity field remains strictly zero, preventing artificial roundoff-driven currents.
    // Max infinity norm of velocity: ||u^n||_inf < 10^{-14} m/s
    double max_velocity_inf_norm = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        ASSERT_FALSE(std::isnan(u[idx]) || std::isinf(u[idx]))
            << "Numerical Instability: Velocity u became NaN or Inf.";
        ASSERT_FALSE(std::isnan(v[idx]) || std::isinf(v[idx]))
            << "Numerical Instability: Velocity v became NaN or Inf.";
        ASSERT_FALSE(std::isnan(w[idx]) || std::isinf(w[idx]))
            << "Numerical Instability: Velocity w became NaN or Inf.";

        double mag = std::max({std::abs(u[idx]), std::abs(v[idx]), std::abs(w[idx])});
        max_velocity_inf_norm = std::max(max_velocity_inf_norm, mag);
    }
    EXPECT_LT(max_velocity_inf_norm, 1e-14) 
        << "Assertion 3 Failed: Artificial velocities induced by floating-point roundoff exceed tolerance (max: " << max_velocity_inf_norm << ").";
}
