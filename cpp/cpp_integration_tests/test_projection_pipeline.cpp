/**
 * @file test_projection_pipeline.cpp
 * @brief Integration test suite verifying Chorin's Projection Method and Helmholtz-Hodge Decomposition.
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
 *         (u* - u^n) / Δt = -(u^n · ∇)u^n + ν ∇^2 u^n + f
 * 
 * Stage 2 (Pressure Poisson Equation Formulation & Solve):
 *     Taking the divergence of the corrector equation yields the scalar Poisson equation:
 *         div u^(n+1) = div u* - (Δt / ρ) ∇^2 p^(n+1)
 *     
 *     Enforcing the solenoidal constraint div u^(n+1) = 0 produces:
 *         ∇^2 p^(n+1) = (ρ / Δt) div u*
 * 
 * Stage 3 (Corrector Step):
 *     Subtract the pressure gradient from the trial velocity field to project it onto the 
 *     divergence-free subspace:
 *         u^(n+1) = u* - (Δt / ρ) grad p^(n+1)
 * 
 * Stage 4 (State Update & Mass Balance Verification):
 *     Verify that max |div u^(n+1)| < ε and update state buffers.
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "orchestrator.hpp"
#include "field_schema.hpp"

using namespace ops;

class ProjectionPipelineTest : public ::testing::Test {
protected:
    // We define physical dynamic parameters and grid configuration for the integration domain.
    // Domain density ρ = 1.0 kg/m^3, dynamic viscosity μ = 0.01 Pa·s, time step Δt = 0.001 s.
    void SetUp() override {
        density = 1.0; 
        dt = 0.001;    
        mu = 0.01;     
        grid_resolution = 32;
    }

    // Evaluates the discrete interior divergence div u using 
    // second-order central spatial differences:
    //     (div u)_(i,j,k) = (u_(i+1,j,k) - u_(i-1,j,k)) / (2 Δx) 
    //                     + (v_(i,j+1,k) - v_(i,j-1,k)) / (2 Δy) 
    //                     + (w_(i,j,k+1) - w_(i,j,k-1)) / (2 Δz)
    double ComputeMaxDivergence(
        const std::vector<double>& u,
        const std::vector<double>& v,
        const std::vector<double>& w,
        const GridDimensions& dims
    ) {
        int nx = dims.nx;
        int ny = dims.ny;
        int nz = dims.nz;
        double inv_2dx = 1.0 / (2.0 * dims.dx);
        double max_div = 0.0;

        // Iterate through interior grid cells (1 to N-2) to avoid boundary artifact skewing
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    size_t idx_px = (i + 1) + nx * (j + ny * k);
                    size_t idx_nx = (i - 1) + nx * (j + ny * k);
                    size_t idx_py = i + nx * ((j + 1) + ny * k);
                    size_t idx_ny = i + nx * ((j - 1) + ny * k);
                    size_t idx_pz = i + nx * (j + ny * (k + 1));
                    size_t idx_nz = i + nx * (j + ny * (k - 1));

                    // Partial derivatives computed via central differences
                    double du_dx = (u[idx_px] - u[idx_nx]) * inv_2dx;
                    double dv_dy = (v[idx_py] - v[idx_ny]) * inv_2dx;
                    double dw_dz = (w[idx_pz] - w[idx_nz]) * inv_2dx;

                    // Absolute divergence: |div u| = |∂u/∂x + ∂v/∂y + ∂w/∂z|
                    double div = std::abs(du_dx + dv_dy + dw_dz);
                    max_div = std::max(max_div, div);
                }
            }
        }
        return max_div;
    }

    double density;
    double dt;
    double mu;
    int grid_resolution;
};

// =================================================================================
// Scenario 1.1: Arbitrary Non-Zero Divergent Field Projection & Mass Conservation
// =================================================================================
TEST_F(ProjectionPipelineTest, NonZeroDivergentFieldCorrection) {
    // -----------------------------------------------------------------------------
    // Step 1: Instantiate domain geometry and solver configuration.
    // We create a uniform 3D cubic domain [0, 1]^3 discretized with 32^3 cells.
    // Spatial mesh spacing Δx = 1.0 / 32 = 0.03125 m.
    // -----------------------------------------------------------------------------
    GridDimensions dims{grid_resolution, grid_resolution, grid_resolution, 1.0 / grid_resolution};
    SolverConfig config;
    config.density = density;

    NavierStokesOrchestrator orchestrator(dims, config);

    // -----------------------------------------------------------------------------
    // Step 2: Allocate dynamic vector buffers for field components.
    // Total cells N = 32^3 = 32,768 cells.
    // -----------------------------------------------------------------------------
    size_t total_cells = static_cast<size_t>(grid_resolution) * grid_resolution * grid_resolution;
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1); // 1 = Fluid cell, 0 = Solid boundary
    std::vector<BoundaryCondition> bc_list;

    // We initialize the velocity field using an analytical vector field:
    //     u(x, y, z) = x
    //     v(x, y, z) = -y + x^2
    //     w(x, y, z) = 0
    //
    // Analytical Divergence Calculation:
    //     div u^0 = ∂u/∂x + ∂v/∂y + ∂w/∂z
    //             = ∂/∂x (x) + ∂/∂y (-y + x^2) + 0
    //             = 1 - 1 + 0 = 0  (only along x = 0)
    //
    // For x > 0:
    //     div u^0 = 2x ≠ 0
    //
    // This provides a controllable, non-zero divergent velocity field across the interior.
    double dx = dims.dx;
    for (int k = 0; k < grid_resolution; ++k) {
        for (int j = 0; j < grid_resolution; ++j) {
            for (int i = 0; i < grid_resolution; ++i) {
                size_t idx = i + grid_resolution * (j + grid_resolution * k);
                double x = i * dx;
                double y = j * dx;
                u[idx] = x * x;
                v[idx] = -y + (x * x);
                w[idx] = 0.0;
            }
        }
    }

    // Preserve baseline copies of u^0 for state update validation
    std::vector<double> u0 = u;
    std::vector<double> v0 = v;

    // Verify baseline state: Ensure the unprojected initial field possesses significant non-zero divergence.
    // We expect max |div u^0| > 10^-4.
    double initial_max_div = ComputeMaxDivergence(u, v, w, dims);
    ASSERT_GT(initial_max_div, 1e-4) 
        << "Pre-condition validation failed: Input vector field u^0 must exhibit non-zero divergence.";

    // -----------------------------------------------------------------------------
    // Step 3: Execute full projection pipeline step via Orchestrator API.
    // Performs: Predictor -> Poisson Source Build -> Linear System Solve -> Corrector.
    // -----------------------------------------------------------------------------
    orchestrator.step(dt, mu, fx, fy, fz, mask, bc_list, u, v, w, p);

    // -----------------------------------------------------------------------------
    // Assertion 1: Verify Non-Trivial Pressure Poisson Solver Execution.
    // Because div u* ≠ 0, the Poisson source term 
    // b = (ρ / Δt) div u* is non-zero, requiring 
    // the Poisson solver to generate a non-trivial pressure scalar field p^(n+1).
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
    // Subtracting the gradient of calculated pressure (u^(n+1) = u* - (Δt / ρ) grad p^(n+1))
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
    // Confirm that the projection operator actively modified the velocity vector components
    // (||u^(n+1) - u^0||_∞ > 0), proving that projection non-trivially
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
