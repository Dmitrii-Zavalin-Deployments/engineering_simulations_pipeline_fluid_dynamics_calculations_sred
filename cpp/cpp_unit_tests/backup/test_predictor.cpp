/**
 * @file test_predictor.cpp
 * @brief Literate Test Suite for Step 1 Predictor Kernel (Trial Velocity Computation)
 *
 * This test file acts as a narrative document. Explanatory text and physical 
 * formulas are written as commented prose using ASCII formatting, while the executable C++ assertions 
 * verify numerical accuracy, contract safety guards, and multi-threading correctness.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>
#include "predictor.hpp"
#include "grid_math.hpp"

namespace navier_stokes_solver {

// ============================================================================
// NARRATIVE SECTION 1: Input Validation and Contract Safety Guards
// ============================================================================
// Before executing floating-point physics kernels, the predictor must rigorously
// defend against invalid memory addresses, degenerate geometries, and illegal 
// physical parameters. Here we verify that contract violations throw immediate exceptions.
//
// Mathematical & Topological Contracts:
//   - Pointer Validity:       u, v, w, fx, fy, fz, u*, v*, w* != nullptr
//   - Dimensionality:         dim(gravity) = 3 [gx, gy, gz]
//   - Topological Conformance: size(mask) = nx * ny * nz
//   - Grid Resolution:        nx, ny, nz >= 3
//   - Spatial Discretization: dx, dy, dz > 0
//   - Temporal Step:          dt > 0
//   - Fluid Properties:       nu >= 0, density > 0
// ============================================================================

TEST(PredictorTest, NullPointerThrowsInvalidArgument) {
    // We define minimal valid grid dimensions and fluid properties.
    GridDimensions dims = {5, 5, 5, 0.1, 0.1, 0.1};
    FluidProperties fluid = {0.01, 1000.0};
    double dt = 0.01;

    // We allocate a valid buffer and mask to test individual pointer invalidations.
    std::vector<double> valid_buffer(125, 1.0);
    std::vector<double> output_buffer(125, 0.0);
    std::vector<double> gravity(3, 0.0);
    std::vector<int> mask(125, 1);

    // If any input pointer (e.g., baseline velocity u) is null, the predictor 
    // must throw an invalid_argument exception to prevent segmentation faults.
    EXPECT_THROW(
        compute_trial_velocities(
            dims, fluid, dt,
            nullptr, valid_buffer.data(), valid_buffer.data(),
            valid_buffer.data(), valid_buffer.data(), valid_buffer.data(),
            gravity,
            mask,
            output_buffer.data(), output_buffer.data(), output_buffer.data()
        ),
        std::invalid_argument
    );
}

TEST(PredictorTest, GravitySizeMismatchThrowsInvalidArgument) {
    // We define minimal valid grid dimensions and fluid properties.
    GridDimensions dims = {5, 5, 5, 0.1, 0.1, 0.1};
    FluidProperties fluid = {0.01, 1000.0};
    double dt = 0.01;

    std::vector<double> valid_buffer(125, 1.0);
    std::vector<double> output_buffer(125, 0.0);
    std::vector<int> mask(125, 1);

    // Rule: The gravity vector must contain exactly 3 components [gx, gy, gz].
    // Supplying a vector of incorrect size (e.g., size 2) violates the physics contract.
    std::vector<double> invalid_gravity = {0.0, -9.81};

    EXPECT_THROW(
        compute_trial_velocities(
            dims, fluid, dt,
            valid_buffer.data(), valid_buffer.data(), valid_buffer.data(),
            valid_buffer.data(), valid_buffer.data(), valid_buffer.data(),
            invalid_gravity,
            mask,
            output_buffer.data(), output_buffer.data(), output_buffer.data()
        ),
        std::invalid_argument
    );
}

TEST(PredictorTest, MaskSizeMismatchThrowsInvalidArgument) {
    // We define minimal valid grid dimensions and fluid properties.
    // Total cells = 5 * 5 * 5 = 125.
    GridDimensions dims = {5, 5, 5, 0.1, 0.1, 0.1};
    FluidProperties fluid = {0.01, 1000.0};
    double dt = 0.01;

    std::vector<double> buf(125, 1.0);
    std::vector<double> out(125, 0.0);
    std::vector<double> gravity(3, 0.0);

    // Rule: A mask vector whose element count differs from the total grid volume 
    // (nx * ny * nz) violates topological consistency and must trigger an exception.
    std::vector<int> invalid_mask(100, 1);

    EXPECT_THROW(
        compute_trial_velocities(
            dims, fluid, dt,
            buf.data(), buf.data(), buf.data(),
            buf.data(), buf.data(), buf.data(),
            gravity,
            invalid_mask,
            out.data(), out.data(), out.data()
        ),
        std::invalid_argument
    );
}

TEST(PredictorTest, InvalidGeometryAndPhysicsParametersThrowErrors) {
    GridDimensions valid_dims = {5, 5, 5, 0.1, 0.1, 0.1};
    FluidProperties valid_fluid = {0.01, 1000.0};
    double valid_dt = 0.01;
    std::vector<double> buf(125, 1.0);
    std::vector<double> out(125, 0.0);
    std::vector<double> gravity(3, 0.0);
    std::vector<int> valid_mask(125, 1);

    // Rule 1: Grid dimensions smaller than 3x3x3 violate central finite difference stencils.
    GridDimensions small_dims = {2, 5, 5, 0.1, 0.1, 0.1};
    std::vector<int> small_mask(50, 1);
    EXPECT_THROW(
        compute_trial_velocities(
            small_dims, valid_fluid, valid_dt,
            buf.data(), buf.data(), buf.data(),
            buf.data(), buf.data(), buf.data(),
            gravity,
            small_mask,
            out.data(), out.data(), out.data()
        ),
        std::invalid_argument
    );

    // Rule 2: Non-positive grid spacing dx, dy, or dz violates spatial discretization rules.
    GridDimensions invalid_dx_dims = {5, 5, 5, 0.0, 0.1, 0.1};
    EXPECT_THROW(
        compute_trial_velocities(
            invalid_dx_dims, valid_fluid, valid_dt,
            buf.data(), buf.data(), buf.data(),
            buf.data(), buf.data(), buf.data(),
            gravity,
            valid_mask,
            out.data(), out.data(), out.data()
        ),
        std::invalid_argument
    );

    // Rule 3: Negative or zero time step dt violates temporal progression rules.
    double invalid_dt = -0.01;
    EXPECT_THROW(
        compute_trial_velocities(
            valid_dims, valid_fluid, invalid_dt,
            buf.data(), buf.data(), buf.data(),
            buf.data(), buf.data(), buf.data(),
            gravity,
            valid_mask,
            out.data(), out.data(), out.data()
        ),
        std::invalid_argument
    );

    // Rule 4: Negative kinematic viscosity nu is physically impossible.
    FluidProperties invalid_nu_fluid = {-0.01, 1000.0};
    EXPECT_THROW(
        compute_trial_velocities(
            valid_dims, invalid_nu_fluid, valid_dt,
            buf.data(), buf.data(), buf.data(),
            buf.data(), buf.data(), buf.data(),
            gravity,
            valid_mask,
            out.data(), out.data(), out.data()
        ),
        std::invalid_argument
    );

    // Rule 5: Non-positive density rho violates continuum physics assumptions.
    FluidProperties invalid_rho_fluid = {0.01, 0.0};
    EXPECT_THROW(
        compute_trial_velocities(
            valid_dims, invalid_rho_fluid, valid_dt,
            buf.data(), buf.data(), buf.data(),
            buf.data(), buf.data(), buf.data(),
            gravity,
            valid_mask,
            out.data(), out.data(), out.data()
        ),
        std::invalid_argument
    );
}

// ============================================================================
// NARRATIVE SECTION 2: Deterministic Baseline Verification (Uniform Flow)
// ============================================================================
// In a perfectly uniform flow field where velocity components are constant everywhere 
// (e.g., u = 2.0, v = 1.0, w = 0.5), spatial gradients (advection and Laplacian) 
// must evaluate precisely to zero:
//     grad(u) = 0,  Laplacian(u) = 0
// Therefore, the explicit Forward-Euler trial velocity update simplifies directly to:
//     u* = u^n + dt * (fx / rho)
// ============================================================================

TEST(PredictorTest, UniformFlowExactEulerUpdate) {
    int nx = 5, ny = 5, nz = 5;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    GridDimensions dims = {nx, ny, nz, 1.0, 1.0, 1.0};
    FluidProperties fluid = {0.01, 1000.0};
    double dt = 0.1;

    std::vector<double> u(total_cells, 2.0);
    std::vector<double> v(total_cells, 1.0);
    std::vector<double> w(total_cells, 0.5);

    // Apply constant external body forces
    std::vector<double> fx(total_cells, 0.5);
    std::vector<double> fy(total_cells, -0.2);
    std::vector<double> fz(total_cells, 0.1);

    std::vector<double> gravity(3, 0.0);
    std::vector<int> mask(total_cells, 1);

    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);

    // Execute the predictor kernel
    compute_trial_velocities(
        dims, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity,
        mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // Check interior cells (since boundary cells are skipped by mask/stencil constraints)
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));

                // Expected calculation accounting for density scaling (rho = 1000.0):
                // u_star = u + dt * (fx / rho) = 2.0 + 0.1 * (0.5 / 1000.0) = 2.00005
                EXPECT_NEAR(u_star[idx], 2.00005, 1e-12);
                // v_star = 1.0 + 0.1 * (-0.2 / 1000.0) = 0.99998
                EXPECT_NEAR(v_star[idx], 0.99998, 1e-12);
                // w_star = 0.5 + 0.1 * (0.1 / 1000.0) = 0.50001
                EXPECT_NEAR(w_star[idx], 0.50001, 1e-12);
            }
        }
    }
}

// ============================================================================
// NARRATIVE SECTION 3: Multi-threading Correctness & Execution Verification
// ============================================================================
// To ensure OpenMP multi-threading executes correctly without data corruption, 
// race conditions, or incorrect chunk distribution, we test a large grid domain 
// (15 x 15 x 15 = 3375 cells), which strictly exceeds our threshold 
// of 1000 cells to trigger parallel thread execution. We verify that all interior 
// cells compute correct deterministic values and remain completely free of NaN/Inf anomalies.
// ============================================================================

TEST(PredictorTest, MultiThreadingParallelExecutionCorrectness) {
    // Grid size 15x15x15 = 3375 cells (> 1000 threshold, activating OpenMP parallel region)
    int nx = 15, ny = 15, nz = 15;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    GridDimensions dims = {nx, ny, nz, 0.5, 0.5, 0.5};
    FluidProperties fluid = {0.02, 1000.0};
    double dt = 0.05;

    std::vector<double> u(total_cells, 1.5);
    std::vector<double> v(total_cells, -0.5);
    std::vector<double> w(total_cells, 1.0);
    std::vector<double> fx(total_cells, 0.1);
    std::vector<double> fy(total_cells, 0.1);
    std::vector<double> fz(total_cells, 0.1);

    std::vector<double> gravity(3, 0.0);
    std::vector<int> mask(total_cells, 1);

    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);

    // Run kernel across multiple threads
    EXPECT_NO_THROW(
        compute_trial_velocities(
            dims, fluid, dt,
            u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(),
            gravity,
            mask,
            u_star.data(), v_star.data(), w_star.data()
        )
    );

    // Verify that every interior cell was processed correctly in parallel
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));

                // With uniform flow, advection and Laplacian are zero. 
                // Expected u_star = 1.5 + dt * (fx / rho) = 1.5 + 0.05 * (0.1 / 1000.0) = 1.500005
                EXPECT_NEAR(u_star[idx], 1.500005, 1e-12);
                // Ensure no thread produced non-finite artifacts
                EXPECT_TRUE(std::isfinite(u_star[idx]));
                EXPECT_TRUE(std::isfinite(v_star[idx]));
                EXPECT_TRUE(std::isfinite(w_star[idx]));
            }
        }
    }
}

// ============================================================================
// NARRATIVE SECTION 4: Robustness and Numerical Exception Safety
// ============================================================================
// Verifies that non-finite floating-point results (such as NaN or Infinity 
// generated during force evaluations or integration steps) trigger the 
// guarded runtime error exception safely across execution threads.
// ============================================================================

TEST(PredictorTest, NonFiniteVelocityThrowsRuntimeError) {
    int nx = 5, ny = 5, nz = 5;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    GridDimensions dims = {nx, ny, nz, 0.1, 0.1, 0.1};
    FluidProperties fluid = {0.01, 1000.0};
    double dt = 0.1;

    std::vector<double> u(total_cells, 1.0);
    std::vector<double> v(total_cells, 1.0);
    std::vector<double> w(total_cells, 1.0);
    
    // Inject a NaN into an interior cell force component to trigger non-finite detection
    std::vector<double> fx(total_cells, 1.0);
    size_t target_idx = static_cast<size_t>(get_flat_index(2, 2, 2, nx, ny)); // interior cell (2,2,2)
    fx[target_idx] = std::numeric_limits<double>::quiet_NaN();

    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);
    std::vector<double> gravity(3, 0.0);
    std::vector<int> mask(total_cells, 1);

    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);

    EXPECT_THROW(
        compute_trial_velocities(
            dims, fluid, dt,
            u.data(), v.data(), w.data(),
            fx.data(), fy.data(), fz.data(),
            gravity,
            mask,
            u_star.data(), v_star.data(), w_star.data()
        ),
        std::runtime_error
    );
}

// ============================================================================
// NARRATIVE SECTION 5: Mask Protection of Non-Fluid (Boundary/Solid) States
// ============================================================================
// Verifies that cells with mask != 1 (such as Dirichlet boundaries or solid walls)
// are strictly preserved and left unmodified by the predictor trial velocity update,
// even when large external body forces or gradients are present across the grid.
// ============================================================================

TEST(PredictorTest, MaskProtectsNonFluidCellsFromModification) {
    int nx = 5, ny = 5, nz = 5;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    GridDimensions dims = {nx, ny, nz, 1.0, 1.0, 1.0};
    FluidProperties fluid = {0.01, 1000.0};
    double dt = 0.1;

    std::vector<double> u(total_cells, 1.0);
    std::vector<double> v(total_cells, 1.0);
    std::vector<double> w(total_cells, 1.0);

    // Apply massive body forces that would aggressively alter active fluid cells
    std::vector<double> fx(total_cells, 50.0);
    std::vector<double> fy(total_cells, 50.0);
    std::vector<double> fz(total_cells, 50.0);

    // Configure mask: default all active fluid (1), but designate specific cells 
    // as solid walls (0) and Dirichlet boundaries (-1).
    std::vector<double> gravity(3, 0.0);
    std::vector<int> mask(total_cells, 1);

    size_t solid_idx = static_cast<size_t>(get_flat_index(2, 2, 2, nx, ny));     // cell (2,2,2)
    size_t dirichlet_idx = static_cast<size_t>(get_flat_index(1, 1, 1, nx, ny));  // cell (1,1,1)

    mask[solid_idx] = 0;      // Solid state
    mask[dirichlet_idx] = -1; // Dirichlet boundary state

    // Pre-define specific static values for these protected cells
    u[solid_idx] = 42.0;
    v[solid_idx] = 42.0;
    w[solid_idx] = 42.0;

    u[dirichlet_idx] = 99.0;
    v[dirichlet_idx] = 99.0;
    w[dirichlet_idx] = 99.0;

    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);

    compute_trial_velocities(
        dims, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity,
        mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // Assert that non-fluid cells (mask != 1) copy their baseline velocity directly
    EXPECT_DOUBLE_EQ(u_star[solid_idx], 42.0);
    EXPECT_DOUBLE_EQ(v_star[solid_idx], 42.0);
    EXPECT_DOUBLE_EQ(w_star[solid_idx], 42.0);

    EXPECT_DOUBLE_EQ(u_star[dirichlet_idx], 99.0);
    EXPECT_DOUBLE_EQ(v_star[dirichlet_idx], 99.0);
    EXPECT_DOUBLE_EQ(w_star[dirichlet_idx], 99.0);

    // Verify preservation across all non-fluid cells
    for (size_t i = 0; i < total_cells; ++i) {
        if (mask[i] != 1) {
            EXPECT_DOUBLE_EQ(u_star[i], u[i]);
            EXPECT_DOUBLE_EQ(v_star[i], v[i]);
            EXPECT_DOUBLE_EQ(w_star[i], w[i]);
        }
    }
}

} // namespace navier_stokes_solver
