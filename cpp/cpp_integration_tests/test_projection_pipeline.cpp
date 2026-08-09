/**
 * @file test_projection_pipeline.cpp
 * @brief Literate Integration Test Suite verifying Chorin's Projection Method 
 *        and Helmholtz-Hodge Decomposition with JSON-driven runtime assets.
 * 
 * LITERATE TESTING NARRATIVE & MATHEMATICAL DERIVATION:
 * ---------------------------------------------------------------------------------
 * The Helmholtz-Hodge Decomposition Theorem states that any smooth vector field u
 * defined on a bounded domain can be uniquely decomposed into a divergence-free (solenoidal) 
 * component u_sol and an irrotational (curl-free) component grad phi:
 * 
 *     u = u_sol + grad phi,  where  div u_sol = 0
 * 
 * In incompressible Navier-Stokes solvers, Chorin's fractional-step projection method uses 
 * this theorem to enforce mass conservation (div u = 0) at each time step.
 * 
 * The physical time step proceeds through four sequential mathematical stages:
 * 
 * Stage 1 (Predictor Step):
 *     Compute an intermediate non-solenoidal velocity field u* by advancing advection,
 *     viscous diffusion, and body forces without the pressure gradient:
 *         (u* - u^n) / dt = -(u^n . grad)u^n + nu * grad^2 u^n + f
 * 
 * Stage 2 (Pressure Poisson Equation Formulation & Solve):
 *     Taking the divergence of the corrector equation yields the scalar Poisson equation:
 *         div u^(n+1) = div u* - (dt / rho) * grad^2 p^(n+1)
 *     
 *     Enforcing the solenoidal constraint div u^(n+1) = 0 produces:
 *         grad^2 p^(n+1) = (rho / dt) * div u*
 * 
 * Stage 3 (Corrector Step):
 *     Subtract the pressure gradient from the trial velocity field to project it onto the 
 *     divergence-free subspace:
 *         u^(n+1) = u* - (dt / rho) * grad p^(n+1)
 * 
 * Stage 4 (State Update & Mass Balance Verification):
 *     Verify that max |div u^(n+1)| < epsilon and update state buffers.
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "orchestrator.hpp"

using json = nlohmann::json;
using namespace ops;

class ProjectionPipelineTest : public ::testing::Test {
protected:
    // Loads domain parameters, physical properties, and execution tolerances 
    // from external JSON configuration files before each test run.
    void SetUp() override {
        std::ifstream config_stream("data/config.json");
        ASSERT_TRUE(config_stream.is_open()) << "Failed to open data/config.json";
        config_stream >> config_json_;

        std::ifstream input_stream("data/navier_stokes_input.json");
        ASSERT_TRUE(input_stream.is_open()) << "Failed to open data/navier_stokes_input.json";
        input_stream >> input_json_;
    }

    // Evaluates the discrete interior divergence div u using second-order central spatial differences:
    //     (div u)_(i,j,k) = (u_(i+1,j,k) - u_(i-1,j,k)) / (2 * dx) 
    //                     + (v_(i,j+1,k) - v_(i,j-1,k)) / (2 * dy) 
    //                     + (w_(i,j,k+1) - w_(i,j,k-1)) / (2 * dz)
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

        // Iterate through interior grid cells (1 to N-2) to avoid boundary artifact skewing
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    size_t idx_px = (i + 1) + static_cast<size_t>(nx) * (j + ny * k);
                    size_t idx_nx = (i - 1) + static_cast<size_t>(nx) * (j + ny * k);
                    size_t idx_py = i + static_cast<size_t>(nx) * ((j + 1) + ny * k);
                    size_t idx_ny = i + static_cast<size_t>(nx) * ((j - 1) + ny * k);
                    size_t idx_pz = i + static_cast<size_t>(nx) * (j + ny * (k + 1));
                    size_t idx_nz = i + static_cast<size_t>(nx) * (j + ny * (k - 1));

                    // Partial derivatives computed via central differences
                    double du_dx = (u[idx_px] - u[idx_nx]) / (2.0 * dx);
                    double dv_dy = (v[idx_py] - v[idx_ny]) / (2.0 * dy);
                    double dw_dz = (w[idx_pz] - w[idx_nz]) / (2.0 * dz);

                    // Absolute divergence: |div u| = |du/dx + dv/dy + dw/dz|
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
// Scenario 1.1: Arbitrary Non-Zero Divergent Field Projection & Mass Conservation
// =================================================================================
TEST_F(ProjectionPipelineTest, NonZeroDivergentFieldCorrection) {
    // -----------------------------------------------------------------------------
    // Step 1: Instantiate domain geometry and solver configuration from JSON inputs.
    // Domain dimensions [x_min, x_max] x [y_min, y_max] x [z_min, z_max] and cell numbers
    // (nx, ny, nz) are extracted directly from navier_stokes_input.json.
    // Spatial mesh spacing is computed as:
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

    // Extract fluid physical properties and simulation parameters
    double density = input_json_["fluid_properties"]["density"];
    double mu = input_json_["fluid_properties"]["viscosity"];
    double dt = input_json_["simulation_parameters"]["time_step"];

    // Execution tolerances read directly from flattened config.json
    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = config_json_["max_poisson_iterations"];
    config.poisson_tolerance = config_json_["poisson_tolerance"];

    NavierStokesOrchestrator orchestrator(dims, config);

    // -----------------------------------------------------------------------------
    // Step 2: Allocate dynamic field vectors and populate boundary conditions.
    // Total domain cells N = nx * ny * nz.
    // -----------------------------------------------------------------------------
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // External force vectors loaded from input JSON
    std::vector<double> fx(total_cells, input_json_["external_forces"]["force_vector"][0]);
    std::vector<double> fy(total_cells, input_json_["external_forces"]["force_vector"][1]);
    std::vector<double> fz(total_cells, input_json_["external_forces"]["force_vector"][2]);

    // Domain fluid/obstacle mask loaded from input JSON (1 = Fluid, 0 = Wall)
    std::vector<int> mask;
    mask.reserve(total_cells);
    for (const auto& val : input_json_["mask"]) {
        mask.push_back(val.get<int>());
    }

    // Boundary conditions extracted from input JSON
    std::vector<BoundaryCondition> bc_list;
    for (const auto& bc_item : input_json_["boundary_conditions"]) {
        BoundaryCondition bc;
        bc.type = BoundaryType::NEUMANN;
        bc.value = bc_item["values"].contains("u") ? bc_item["values"]["u"].get<double>() : 0.0;
        bc_list.push_back(bc);
    }

    // We initialize the velocity field using an analytical non-solenoidal profile:
    //     u(x, y, z) = x^2
    //     v(x, y, z) = -y + x^2
    //     w(x, y, z) = 0
    //
    // Analytical Divergence Calculation:
    //     div u^0 = du/dx + dv/dy + dw/dz
    //             = d/dx(x^2) + d/dy(-y + x^2) + 0
    //             = 2x - 1
    //
    // For x != 0.5:
    //     div u^0 = 2x - 1 != 0
    //
    // This establishes a controllable, non-zero divergent velocity field across the interior domain.
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

    // Preserve baseline initial state vector u^0 for state modification verification
    std::vector<double> u0 = u;
    std::vector<double> v0 = v;

    // Verify baseline state: Ensure initial vector field exhibits significant non-zero divergence
    // Expect max |div u^0| > 10^-4.
    double initial_max_div = ComputeMaxDivergence(u, v, w, dims);
    ASSERT_GT(initial_max_div, 1e-4) 
        << "Pre-condition validation failed: Input vector field u^0 must exhibit non-zero divergence.";

    // -----------------------------------------------------------------------------
    // Step 3: Execute full projection step via Orchestrator API.
    // Performs: Predictor -> Poisson Source Build -> Linear System Solve -> Corrector.
    // -----------------------------------------------------------------------------
    orchestrator.step(dt, mu, fx, fy, fz, mask, bc_list, u, v, w, p);

    // -----------------------------------------------------------------------------
    // Assertion 1: Verify Non-Trivial Pressure Poisson Solver Execution.
    // Because div u* != 0, the Poisson source term b = (rho / dt) * div u* is non-zero,
    // requiring the Poisson solver to compute a non-trivial pressure scalar field p^(n+1):
    //     max |p^(n+1)| > 0
    // -----------------------------------------------------------------------------
    double max_pressure = 0.0;
    for (double p_val : p) {
        max_pressure = std::max(max_pressure, std::abs(p_val));
    }
    EXPECT_GT(max_pressure, 0.0) 
        << "Assertion 1 Failed: Pressure Poisson solver returned zero pressure everywhere (p^(n+1) = 0).";

    // -----------------------------------------------------------------------------
    // Assertion 2: Divergence Reduction & Mass Conservation.
    // Subtracting the gradient of calculated pressure:
    //     u^(n+1) = u* - (dt / rho) * grad p^(n+1)
    // projects velocity onto the solenoidal manifold, strictly reducing divergence:
    //     max |div u^(n+1)| < max |div u^0|
    // -----------------------------------------------------------------------------
    double final_max_div = ComputeMaxDivergence(u, v, w, dims);
    EXPECT_LT(final_max_div, initial_max_div)
        << "Assertion 2 Failed: Helmholtz-Hodge projection failed to reduce velocity field divergence.";

    // -----------------------------------------------------------------------------
    // Assertion 3: Strict Solenoidal Numerical Threshold.
    // The post-correction velocity field u^(n+1) must satisfy mass conservation
    // within strict numerical tolerances (max |div u^(n+1)| < 10^-4).
    // -----------------------------------------------------------------------------
    EXPECT_LT(final_max_div, 1e-4)
        << "Assertion 3 Failed: Post-correction velocity field fails strict solenoidal tolerance threshold (< 10^-4).";

    // -----------------------------------------------------------------------------
    // Assertion 4: Non-Trivial Field Correction State Update.
    // Confirm that the projection operator actively modified velocity vector components
    // (||u^(n+1) - u^0||_inf > 0), proving that projection non-trivially
    // shifted the initial vector field onto the solenoidal subspace.
    // -----------------------------------------------------------------------------
    double max_field_change = 0.0;
    for (size_t i = 0; i < total_cells; ++i) {
        double delta_u = std::abs(u[i] - u0[i]);
        double delta_v = std::abs(v[i] - v0[i]);
        max_field_change = std::max({max_field_change, delta_u, delta_v});
    }
    EXPECT_GT(max_field_change, 0.0)
        << "Assertion 4 Failed: Velocity buffer was not modified by the projection step.";
}
