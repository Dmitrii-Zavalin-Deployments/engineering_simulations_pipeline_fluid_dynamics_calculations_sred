/**
 * @file test_projection_pipeline.cpp
 * @brief Integration test suite verifying Chorin's Projection Method and Helmholtz-Hodge Decomposition.
 * 
 * LITERATE TESTING NARRATIVE:
 * ---------------------------------------------------------------------------------
 * According to the Helmholtz-Hodge decomposition theorem, any vector field can be uniquely 
 * expressed as the sum of a divergence-free vector field and the gradient of a scalar field.
 * In incompressible fluid dynamics, Chorin's projection method leverages this principle 
 * to enforce mass conservation ($\nabla \cdot \mathbf{u} = 0$) by splitting the momentum 
 * equation into two sequential phases:
 * 
 * 1. PREDICTOR STEP: Advance the velocity field using advection, diffusion, and body forces 
 *    to compute an intermediate, non-divergence-free trial velocity field u*.
 * 
 * 2. PRESSURE POISON SOLVE: Formulate and solve the Poisson equation for pressure using 
 *    the divergence of the trial velocity as the source term:
 *        \nabla^2 p^{n+1} = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*
 * 
 * 3. CORRECTOR STEP: Project the trial velocity onto the divergence-free subspace by 
 *    subtracting the scaled pressure gradient:
 *        \mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla p^{n+1}
 * 
 * 4. STATE SWAP: Advance the internal state pointer so that \mathbf{u}^{n+1} becomes 
 *    \mathbf{u}^n for the subsequent time step.
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include "orchestrator.hpp"
#include "field_schema.hpp"

// We define a test fixture for the Projection Pipeline to configure grid parameters,
// fluid density, and temporal stepping characteristics prior to execution.
class ProjectionPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize simulation grid dimensions and physical parameters.
        density = 1.0; // kg/m^3
        dt = 0.001;    // s
        grid_resolution = 32;
    }

    double density;
    double dt;
    int grid_resolution;
};

// Scenario 1.1: Arbitrary Non-Zero Divergent Field Correction
TEST_F(ProjectionPipelineTest, NonZeroDivergentFieldCorrection) {
    // -----------------------------------------------------------------------------
    // Step 1: Initialize grid dimensions and configuration structures for NavierStokesOrchestrator.
    // -----------------------------------------------------------------------------
    GridDimensions dims{grid_resolution, grid_resolution, grid_resolution, 1.0 / grid_resolution};
    SolverConfig config;
    config.density = density;

    NavierStokesOrchestrator orchestrator(dims, config);

    // -----------------------------------------------------------------------------
    // Step 2: Construct an initial velocity field explicitly containing non-zero divergence.
    // We choose the analytical field: u0 = (x, -y + x^2)
    // The mathematical divergence of this field is:
    //     \nabla \cdot \mathbf{u}^0 = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 1 + (-1) + 0 = \text{non-zero function}
    // Specifically, for 2D/3D components: \nabla \cdot \mathbf{u}^0 = 1 + 2x \neq 0.
    // -----------------------------------------------------------------------------
    orchestrator.InitializeCustomVelocityField([](double x, double y, double z, double& u, double& v, double& w) {
        u = x;
        v = -y + (x * x);
        w = 0.0;
    });

    // Capture the initial maximum divergence magnitude to verify baseline non-zero state.
    double initial_max_div = orchestrator.ComputeMaxDivergence();
    ASSERT_GT(initial_max_div, 1e-4) << "Baseline setup failed: Initial field must possess non-zero divergence.";

    // -----------------------------------------------------------------------------
    // Step 3: Execute a single integrated time step via the orchestrator pipeline.
    // This internally triggers the Predictor, Poisson Setup, Poisson Solve, and Corrector.
    // -----------------------------------------------------------------------------
    orchestrator.SolveSingleTimeStep(dt);

    // -----------------------------------------------------------------------------
    // Assertion 1: Intermediate trial velocity u* retains non-zero divergence.
    // During the predictor step, explicit body and convective forces are applied, 
    // but mass conservation is intentionally ignored until the projection step.
    // -----------------------------------------------------------------------------
    double trial_max_div = orchestrator.GetTrialVelocityDivergenceMax();
    EXPECT_NE(trial_max_div, 0.0) 
        << "Assertion 1 Failed: Trial velocity u* erroneously exhibits zero divergence prematurely.";

    // -----------------------------------------------------------------------------
    // Assertion 2: Right-hand side vector b is correctly populated.
    // The source term for the pressure Poisson equation is defined as:
    //     b = (\rho / \Delta t) * (\nabla \cdot \mathbf{u}^*)
    // We verify that the L2 norm of the populated right-hand side vector scales 
    // proportionally with the trial velocity divergence.
    // -----------------------------------------------------------------------------
    double rhs_norm = orchestrator.GetPoissonRHSNorm();
    double expected_rhs_approx = (density / dt) * trial_max_div;
    EXPECT_NEAR(rhs_norm, expected_rhs_approx, 0.1)
        << "Assertion 2 Failed: Poisson right-hand side vector 'b' was improperly populated.";

    // -----------------------------------------------------------------------------
    // Assertion 3: Post-correction velocity satisfies mass conservation.
    // After subtracting the pressure gradient, the final corrected velocity field 
    // must be divergence-free within strict numerical tolerances:
    //     \max | \nabla \cdot \mathbf{u}^1 | < 10^{-6}
    // -----------------------------------------------------------------------------
    double final_max_div = orchestrator.ComputeMaxDivergence();
    EXPECT_LT(final_max_div, 1e-6)
        << "Assertion 3 Failed: Post-correction velocity field fails mass conservation threshold (< 10^-6).";

    // -----------------------------------------------------------------------------
    // Assertion 4: State swap validation.
    // Confirm that the newly computed divergence-free velocity field \mathbf{u}^1 
    // has been safely swapped into the active working buffer \mathbf{u}^n for the next iteration.
    // -----------------------------------------------------------------------------
    bool is_state_swapped = orchestrator.VerifyActiveStateBufferUpdated();
    EXPECT_TRUE(is_state_swapped)
        << "Assertion 4 Failed: State swap procedure failed to update active step pointers.";
}
