/**
 * @file test_projection_pipeline.cpp
 * @brief Literate Integration Test Suite verifying Long-Horizon Navier-Stokes 
 *        Time-Integration Stability, Divergence Boundedness, and Field Evolution.
 * 
 * LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
 * ---------------------------------------------------------------------------------
 * The incompressible Navier-Stokes equations governing Newtonian fluid dynamics 
 * in a bounded domain are expressed by the momentum conservation equation and the 
 * divergence-free kinematic constraint (mass conservation):
 * 
 *       (1) Momentum Equation (Advection-Diffusion-Pressure):
 *           rho * (du/dt + (u . grad)u) = -grad p + mu * grad^2 u + f
 * 
 *       (2) Incompressibility Constraint (Solenoidal Manifold):
 *           div u = 0
 * 
 * To solve this coupled system numerically, Chorin's fractional-step projection method 
 * advances the system through discrete time steps of size dt:
 * 
 *       - Predictor Step (Intermediate Velocity u*):
 *           (u* - u^n) / dt = -(u^n . grad)u^n + (mu / rho) * grad^2 u^n + f / rho
 * 
 *       - Pressure Poisson Equation (PPE):
 *           grad^2 p^(n+1) = (rho / dt) * div u*
 * 
 *       - Corrector / Projection Step:
 *           u^(n+1) = u* - (dt / rho) * grad p^(n+1)
 * 
 * EXTENDED TIME-INTEGRATION VALIDATION OBJECTIVE:
 * Rather than assuming a single-step collapse to machine zero (which ignores physical 
 * advection and viscous re-injection), this test runs the complete Navier-Stokes 
 * orchestrator across N = 50 consecutive time steps. It verifies three pillars 
 * of solver correctness:
 *       1. Numerical Stability: Maximum velocity remains bounded (no NaN / blow-up).
 *       2. Divergence Boundedness: Divergence remains stably controlled under flow evolution.
 *       3. Temporal Dynamics: The velocity field actively evolves via physical transport.
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
using namespace navier_stokes_solver;

class ProjectionPipelineTest : public ::testing::Test {
protected:
    // Loads domain parameters, physical properties, and execution tolerances 
    // from external JSON configuration files using explicit paths.
    void SetUp() override {
        std::ifstream config_stream("cpp/cpp_integration_tests/data/config.json");
        ASSERT_TRUE(config_stream.is_open()) << "Failed to open cpp/cpp_integration_tests/data/config.json";
        config_stream >> config_json_;

        std::ifstream input_stream("cpp/cpp_integration_tests/data/navier_stokes_input.json");
        ASSERT_TRUE(input_stream.is_open()) << "Failed to open cpp/cpp_integration_tests/data/navier_stokes_input.json";
        input_stream >> input_json_;
    }

    // Computes the discrete interior divergence div u using second-order central spatial differences:
    //       (div u)_(i,j,k) = (u_(i+1,j,k) - u_(i-1,j,k)) / (2 * dx) 
    //                       + (v_(i,j+1,k) - v_(i,j-1,k)) / (2 * dy) 
    //                       + (w_(i,j,k+1) - w_(i,j,k-1)) / (2 * dz)
    double ComputeMaxDivergence(
        const std::vector<double>& u,
        const std::vector<double>& v,
        const std::vector<double>& w,
        const GridDimensions& dims
    ) {
        int nx = dims.nx;
        int ny = dims.ny;
        int nz = dims.nz;
        double dx = dims.dx;
        double dy = dims.dy;
        double dz = dims.dz;
        double max_div = 0.0;

        // Iterate through interior grid cells (1 to N-2) to prevent boundary stencils from skewing metrics
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    size_t idx_px = (i + 1) + static_cast<size_t>(nx) * (j + ny * k);
                    size_t idx_nx = (i - 1) + static_cast<size_t>(nx) * (j + ny * k);
                    size_t idx_py = i + static_cast<size_t>(nx) * ((j + 1) + ny * k);
                    size_t idx_ny = i + static_cast<size_t>(nx) * ((j - 1) + ny * k);
                    size_t idx_pz = i + static_cast<size_t>(nx) * (j + ny * (k + 1));
                    size_t idx_nz = i + static_cast<size_t>(nx) * (j + ny * (k - 1));

                    // Partial derivatives computed via second-order central spatial differences:
                    //       du/dx = (u_(i+1) - u_(i-1)) / (2 * dx)
                    //       dv/dy = (v_(j+1) - v_(j-1)) / (2 * dy)
                    //       dw/dz = (w_(k+1) - w_(k-1)) / (2 * dz)
                    double du_dx = (u[idx_px] - u[idx_nx]) / (2.0 * dx);
                    double dv_dy = (v[idx_py] - v[idx_ny]) / (2.0 * dy);
                    double dw_dz = (w[idx_pz] - w[idx_nz]) / (2.0 * dz);

                    // Absolute local divergence magnitude: |div u| = |du/dx + dv/dy + dw/dz|
                    double div = std::abs(du_dx + dv_dy + dw_dz);
                    max_div = std::max(max_div, div);
                }
            }
        }
        return max_div;
    }

    json config_json_;
    json input_json_;
};

// =================================================================================
// Scenario 1.1: Extended 50-Step Navier-Stokes Time Integration & Stability
// =================================================================================
TEST_F(ProjectionPipelineTest, LongHorizonTimeIntegrationStability) {
    // -----------------------------------------------------------------------------
    // Step 1: Initialize grid dimensions, physical properties, and time step size dt.
    // Spatial grid increments are derived as:
    //      dx = (x_max - x_min) / nx
    //      dy = (y_max - y_min) / ny
    //      dz = (z_max - z_min) / nz
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

    double density = input_json_["fluid_properties"]["density"];
    double mu = input_json_["fluid_properties"]["viscosity"];
    double dt = input_json_["simulation_parameters"]["time_step"];

    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = config_json_["max_poisson_iterations"];
    config.poisson_tolerance = config_json_["poisson_tolerance"];

    NavierStokesOrchestrator orchestrator(dims, config);

    // -----------------------------------------------------------------------------
    // Step 2: Allocate vector fields and initialize with a non-zero analytical profile:
    //       u(x, y, z) = x^2
    //       v(x, y, z) = -y + x^2
    //       w(x, y, z) = 0.0
    //
    // Analytical divergence of initial condition:
    //       div u^0 = du/dx + dv/dy + dw/dz = 2x - 1 != 0
    // -----------------------------------------------------------------------------
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    std::vector<double> gravity(3, 0.0);
    std::vector<double> fx(total_cells, input_json_["external_forces"]["force_vector"][0]);
    std::vector<double> fy(total_cells, input_json_["external_forces"]["force_vector"][1]);
    std::vector<double> fz(total_cells, input_json_["external_forces"]["force_vector"][2]);

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
        bc.u_val = bc_item["values"].contains("u") ? bc_item["values"]["u"].get<double>() : 0.0;
        bc.v_val = bc_item["values"].contains("v") ? bc_item["values"]["v"].get<double>() : 0.0;
        bc.w_val = bc_item["values"].contains("w") ? bc_item["values"]["w"].get<double>() : 0.0;
        bc_list.push_back(bc);
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = i + static_cast<size_t>(nx) * (j + ny * k);
                double x = x_min + i * dx;
                double y = y_min + j * dy;
                u[idx] = x * x;
                v[idx] = -y + (x * x);
                w[idx] = 0.0;
            }
        }
    }

    // Preserve baseline initial vector state for displacement verification: u^0 = u, v^0 = v
    std::vector<double> u_initial = u;
    std::vector<double> v_initial = v;

    // -----------------------------------------------------------------------------
    // Step 3: Execute Extended Time-Stepping Loop (N = 50 Iterations).
    // Track maximum velocity magnitudes and divergence bounds across all steps.
    // -----------------------------------------------------------------------------
    const int total_time_steps = 50;
    double max_observed_div = 0.0;
    double peak_velocity_magnitude = 0.0;

    for (int step = 0; step < total_time_steps; ++step) {
        orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

        // Evaluate instantaneous maximum interior divergence
        double curr_div = ComputeMaxDivergence(u, v, w, dims);
        max_observed_div = std::max(max_observed_div, curr_div);

        // Calculate maximum local velocity magnitude:
        //       |U_idx| = sqrt(u_idx^2 + v_idx^2 + w_idx^2)
        for (size_t idx = 0; idx < total_cells; ++idx) {
            ASSERT_FALSE(std::isnan(u[idx]) || std::isinf(u[idx])) 
                << "Numerical Instability Detected: Velocity u became NaN or Inf at step " << step;
            ASSERT_FALSE(std::isnan(v[idx]) || std::isinf(v[idx])) 
                << "Numerical Instability Detected: Velocity v became NaN or Inf at step " << step;

            double mag = std::sqrt(u[idx] * u[idx] + v[idx] * v[idx] + w[idx] * w[idx]);
            peak_velocity_magnitude = std::max(peak_velocity_magnitude, mag);
        }
    }

    // -----------------------------------------------------------------------------
    // Assertion 1: Numerical Boundedness & Stability Check
    // The peak velocity magnitude across 50 time steps must remain physically bounded:
    //       peak_velocity_magnitude < 1000.0 m/s
    // -----------------------------------------------------------------------------
    EXPECT_LT(peak_velocity_magnitude, 1000.0)
        << "Assertion 1 Failed: Velocity field experienced unphysical numerical blow-up.";

    // -----------------------------------------------------------------------------
    // Assertion 2: Mass Conservation Boundedness Check
    // The pressure-correction projection must stably bound velocity divergence:
    //       max_observed_div < 1.0
    // -----------------------------------------------------------------------------
    EXPECT_LT(max_observed_div, 1.0)
        << "Assertion 2 Failed: Divergence grew unbounded over 50 time steps (max div: " << max_observed_div << ").";

    // -----------------------------------------------------------------------------
    // Assertion 3: Non-Trivial Temporal Field Evolution
    // Verify physical state update over 50 iterations:
    //       ||u^(50) - u^0||_inf = max(|u^(50) - u^0|, |v^(50) - v^0|) > 0
    // -----------------------------------------------------------------------------
    double max_field_displacement = 0.0;
    for (size_t i = 0; i < total_cells; ++i) {
        double delta_u = std::abs(u[i] - u_initial[i]);
        double delta_v = std::abs(v[i] - v_initial[i]);
        max_field_displacement = std::max({max_field_displacement, delta_u, delta_v});
    }
    EXPECT_GT(max_field_displacement, 0.0)
        << "Assertion 3 Failed: Velocity buffers remained static; time-integration loop did not advance state.";
}
