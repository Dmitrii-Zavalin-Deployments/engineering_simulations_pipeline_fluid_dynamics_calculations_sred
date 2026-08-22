/**
 * @file test_full_pipeline_literate.cpp
 * @brief Literate-style integration test for the full Navier–Stokes solver pipeline.
 *
 * This test is written as a narrative. Each section explains the physics,
 * the numerical expectations, and the solver responsibilities. Executable
 * assertions appear inline with the prose.
 *
 * ============================================================================
 * SECTION 0 — Test Scenario Overview
 * ============================================================================
 *
 * We simulate a 3D domain:
 *
 *     x ∈ [0, 4],  nx = 8
 *     y ∈ [0, 4],  ny = 8
 *     z ∈ [0, 2],  nz = 4
 *
 * The mask defines an internal fluid cavity surrounded by solid walls (-1)
 * and solid interior obstacles (0). Fluid cells are marked with mask == 1.
 *
 * Boundary conditions:
 *   - z_min: inflow  (w = +1)
 *   - z_max: outflow (w = +1)
 *   - wall:  no-slip (u = v = w = 0)
 *
 * Fluid properties:
 *   - density ρ = 1.0
 *   - viscosity μ = 0.01
 *
 * Simulation parameters:
 *   - dt = 0.1
 *   - total_time = 1.0
 *   - output_interval = 5
 *
 * Solver config:
 *   - max_poisson_iterations = 2000
 *   - poisson_tolerance = 1e-8
 *
 * We will step through the solver pipeline *manually*, asserting correctness
 * after each stage.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>

#include "orchestrator.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"
#include "simulation_prestep.hpp"
#include "predictor.hpp"
#include "pressure_poisson_solver.hpp"
#include "corrector.hpp"
#include "ghost_handler.hpp"

namespace navier_stokes_solver {

TEST(FullPipelineLiterateTest, StepByStepMicroManaged) {

    // ============================================================================
    // SECTION 1 — Grid Setup
    // ============================================================================

    /**
    * The JSON domain is:
    *   x ∈ [0, 4], nx = 8
    *   y ∈ [0, 4], ny = 8
    *   z ∈ [0, 2], nz = 4
    *
    * Python ingestion stores BOTH:
    *   - physical extents (x_min, x_max, ...)
    *   - grid resolution (nx, ny, nz)
    *   - spacing (dx, dy, dz) computed as:
    *
    *       dx = (x_max - x_min) / nx
    *       dy = (y_max - y_min) / ny
    *       dz = (z_max - z_min) / nz
    *
    * The C++ GridDimensions struct stores ONLY:
    *   nx, ny, nz,
    *   dx, dy, dz.
    *
    * Therefore, this test must compute dx, dy, dz using the JSON extents,
    * but must NOT assign x_min/x_max/etc. to dims (they do not exist in C++).
    */

    GridDimensions dims;

    // Physical extents from JSON (local variables only)
    double x_min = 0.0;
    double x_max = 4.0;
    double y_min = 0.0;
    double y_max = 4.0;
    double z_min = 0.0;
    double z_max = 2.0;

    // Resolution from JSON
    dims.nx = 8;
    dims.ny = 8;
    dims.nz = 4;

    // Python-style spacing (division by N, not N−1)
    dims.dx = (x_max - x_min) / dims.nx;
    dims.dy = (y_max - y_min) / dims.ny;
    dims.dz = (z_max - z_min) / dims.nz;

    dims.validate();

    const size_t total_cells = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;

    // ============================================================================
    // SECTION 2 — Allocate Fields
    // ============================================================================

    // All fields begin at zero.
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // Trial velocities (predictor output)
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);

    std::vector<int> mask = {

        // --- Layer k = 0 ---
        0,  0,  0,  0,  0,  0,  0,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0,  0,  0,  0,  0,  0,  0,  0,

        // --- Layer k = 1 ---
        0,  0,  0,  0,  0,  0,  0,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0,  0,  0,  0,  0,  0,  0,  0,

        // --- Layer k = 2 ---
        0,  0,  0,  0,  0,  0,  0,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0,  0,  0,  0,  0,  0,  0,  0,

        // --- Layer k = 3 ---
        0,  0,  0,  0,  0,  0,  0,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    };

    ASSERT_EQ(mask.size(), total_cells);

    // ============================================================================
    // SECTION 3 — Boundary Conditions
    // ============================================================================

    std::vector<BoundaryCondition> bc_list;

    // Inflow at z_min
    {
        BoundaryCondition bc;
        bc.location = "z_min";
        bc.type = "inflow";
        bc.values.has_w = true; bc.values.w = 1.0;
        bc.values.has_u = true; bc.values.u = 0.0;
        bc.values.has_v = true; bc.values.v = 0.0;
        bc.values.has_p = true; bc.values.p = 0.0;
        bc_list.push_back(bc);
    }

    // Outflow at z_max
    {
        BoundaryCondition bc;
        bc.location = "z_max";
        bc.type = "outflow";
        bc.values.has_w = true; bc.values.w = 1.0;
        bc.values.has_u = true; bc.values.u = 0.0;
        bc.values.has_v = true; bc.values.v = 0.0;
        bc.values.has_p = true; bc.values.p = 0.0;
        bc_list.push_back(bc);
    }

    // No-slip walls
    {
        BoundaryCondition bc;
        bc.location = "wall";
        bc.type = "no-slip";
        bc.values.has_u = true; bc.values.u = 0.0;
        bc.values.has_v = true; bc.values.v = 0.0;
        bc.values.has_w = true; bc.values.w = 0.0;
        bc.values.has_p = true; bc.values.p = 0.0;
        bc_list.push_back(bc);
    }

    // ============================================================================
    // SECTION 4 — Solver Configuration
    // ============================================================================

    SolverConfig config;
    config.max_poisson_iterations = 2000;
    config.poisson_tolerance = 1e-8;
    config.density = 1.0;

    const double dt = 0.1;
    const double mu = 0.01;

    std::vector<double> gravity = {0.0, 0.0, 0.0};
    std::vector<double> fx = {0.0, 0.0, 0.0};
    std::vector<double> fy = {0.0, 0.0, 0.0};
    std::vector<double> fz = {0.0, 0.0, 0.0};

    // ============================================================================
    // SECTION 5 — Instantiate Orchestrator
    // ============================================================================

    /**
    * The orchestrator binds together:
    *   - grid dimensions (nx, ny, nz, dx, dy, dz)
    *   - solver configuration (Poisson iterations, tolerance, density)
    *   - internal working buffers (u_star, v_star, w_star, rhs)
    *
    * This constructor does NOT perform any physics.
    * It only allocates persistent buffers sized to total_cells.
    *
    * In test builds (when NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS is enabled),
    * the orchestrator will also record internal snapshots during step() so that
    * the literate test can compare manual stage-by-stage results with the actual
    * orchestrator pipeline. This has no effect in production builds.
    *
    * No assertions are needed here — correctness is validated in subsequent
    * solver stages (pre-step, predictor, PPE, corrector, ghost sync).
    */

    NavierStokesOrchestrator orchestrator(dims, config);

    // ============================================================================
    // SECTION 6 — STEP 1: Pre-Step
    // ============================================================================

    /**
    * In the pre-step, boundary conditions are applied:
    *   - inflow/outflow on z-min/z-max
    *   - no-slip on walls (mask == -1)
    *   - solid interior cells (mask == 0) are clamped
    *
    * EXPECTATIONS (guaranteed by execute_pre_step):
    *   - All mask == -1 cells have u = v = w = 0  (explicit wall BC)
    *   - All mask == 0 cells have u = v = w = 0  (solid interior)
    *   - z_min inflow plane has w = +1
    *   - z_max outflow plane has w = +1
    *   - No NaNs or Infs introduced
    */

    execute_pre_step(u, v, w, p, mask, bc_list, dims.nx, dims.ny, dims.nz);

    // --- Assert pre-step invariants ---
    for (int k = 0; k < dims.nz; ++k) {
        for (int j = 0; j < dims.ny; ++j) {
            for (int i = 0; i < dims.nx; ++i) {

                const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, dims.nx, dims.ny));

                // 1. Wall cells (mask == -1) must be clamped to no-slip
                if (mask[idx] == -1) {
                    ASSERT_NEAR(u[idx], 0.0, 1e-12);
                    ASSERT_NEAR(v[idx], 0.0, 1e-12);
                    ASSERT_NEAR(w[idx], 0.0, 1e-12);
                }

                // 2. Solid interior cells (mask == 0) must also be clamped
                if (mask[idx] == 0) {
                    ASSERT_NEAR(u[idx], 0.0, 1e-12);
                    ASSERT_NEAR(v[idx], 0.0, 1e-12);
                    ASSERT_NEAR(w[idx], 0.0, 1e-12);
                }

                // 3. Inflow plane (z_min)
                if (k == 0 && mask[idx] == 1) {
                    ASSERT_NEAR(w[idx], 1.0, 1e-12);
                }

                // 4. Outflow plane (z_max)
                if (k == dims.nz - 1 && mask[idx] == 1) {
                    ASSERT_NEAR(w[idx], 1.0, 1e-12);
                }

                // 5. No NaNs or Infs anywhere
                ASSERT_TRUE(std::isfinite(u[idx]));
                ASSERT_TRUE(std::isfinite(v[idx]));
                ASSERT_TRUE(std::isfinite(w[idx]));
                ASSERT_TRUE(std::isfinite(p[idx]));
            }
        }
    }

    // ============================================================================
    // SECTION 7 — STEP 2: Predictor
    // ============================================================================

    /**
    * Predictor computes trial velocities u*, v*, w* using:
    *   - advection
    *   - diffusion
    *   - body forces
    *   - gravity
    *
    * EXPECTATIONS (guaranteed by compute_trial_velocities):
    *   - Fluid cells (mask == 1) update normally
    *   - Solid/wall cells (mask != 1) remain clamped to pre-step values
    *   - No NaNs or infinities appear in u*, v*, w*
    *   - Pre-step boundary values are preserved because predictor copies
    *     u, v, w → u*, v*, w* before applying updates only to fluid cells.
    */

    FluidProperties fluid;
    fluid.nu = mu / config.density;   // kinematic viscosity
    fluid.density = config.density;

    // Run predictor
    compute_trial_velocities(
        dims,
        fluid,
        dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity,
        mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // --- Assert predictor invariants ---
    for (int k = 0; k < dims.nz; ++k) {
        for (int j = 0; j < dims.ny; ++j) {
            for (int i = 0; i < dims.nx; ++i) {

                const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, dims.nx, dims.ny));

                // 1. No NaNs or Infs anywhere
                ASSERT_TRUE(std::isfinite(u_star[idx]));
                ASSERT_TRUE(std::isfinite(v_star[idx]));
                ASSERT_TRUE(std::isfinite(w_star[idx]));

                // 2. Solid interior or wall cells must remain clamped
                if (mask[idx] != 1) {
                    ASSERT_NEAR(u_star[idx], u[idx], 1e-12);
                    ASSERT_NEAR(v_star[idx], v[idx], 1e-12);
                    ASSERT_NEAR(w_star[idx], w[idx], 1e-12);
                }

                // 3. Fluid cells should have updated values (not necessarily non-zero)
                if (mask[idx] == 1) {
                    // Predictor guarantees finite values, but not specific magnitudes.
                    ASSERT_TRUE(std::isfinite(u_star[idx]));
                    ASSERT_TRUE(std::isfinite(v_star[idx]));
                    ASSERT_TRUE(std::isfinite(w_star[idx]));
                }
            }
        }
    }

    // ============================================================================
    // SECTION 8 — STEP 3: Poisson Solver
    // ============================================================================

    /**
     * PPE computes pressure that enforces divergence-free velocity.
     *
     * EXPECTATIONS:
     *   - Pressure remains finite
     *   - Solid cells follow Neumann constraints
     *   - Divergence decreases
     */

    // PLACEHOLDER: call PPE
    // solve_pressure_poisson_parallel(p, u_star, v_star, w_star, mask, dims.nx, dims.ny, dims.nz, dt, config.max_poisson_iterations, config.poisson_tolerance);

    // PLACEHOLDER: assert PPE invariants
    // for (size_t idx = 0; idx < total_cells; ++idx) {
    //     ASSERT_TRUE(std::isfinite(p[idx]));
    // }

    // ============================================================================
    // SECTION 9 — STEP 4: Corrector
    // ============================================================================

    /**
     * Corrector projects trial velocities onto divergence-free space:
     *     u = u* - (dt/ρ) ∇p
     *
     * EXPECTATIONS:
     *   - No-penetration at solid interfaces
     *   - No-slip at walls
     *   - Divergence-free interior
     */

    // PLACEHOLDER: call corrector
    // solve_corrector_parallel(u, v, w, u_star, v_star, w_star, p, mask, dims.nx, dims.ny, dims.nz, dt, config.density);

    // PLACEHOLDER: assert corrector invariants
    // for (size_t idx = 0; idx < total_cells; ++idx) {
    //     if (mask[idx] != 1) {
    //         ASSERT_NEAR(u[idx], 0.0, 1e-12);
    //         ASSERT_NEAR(v[idx], 0.0, 1e-12);
    //         ASSERT_NEAR(w[idx], 0.0, 1e-12);
    //     }
    // }

    // ============================================================================
    // SECTION 10 — STEP 5: Ghost Sync
    // ============================================================================

    /**
     * Ghost handler synchronizes buffers and ensures boundary consistency.
     *
     * EXPECTATIONS:
     *   - No overwriting of clamped values
     *   - No NaNs introduced
     */

    // PLACEHOLDER: call ghost sync
    // sync_ghost_buffers(u, v, w, u_star, v_star, w_star, p, mask, dims.nx, dims.ny, dims.nz);

    // ============================================================================
    // SECTION 11 — Final Assertions
    // ============================================================================

    /**
     * At the end of the pipeline:
     *   - All fields must be finite
     *   - All solid/wall cells must be clamped
     *   - Fluid interior must be divergence-free (placeholder)
     */

    for (size_t idx = 0; idx < total_cells; ++idx) {
        ASSERT_TRUE(std::isfinite(u[idx]));
        ASSERT_TRUE(std::isfinite(v[idx]));
        ASSERT_TRUE(std::isfinite(w[idx]));
        ASSERT_TRUE(std::isfinite(p[idx]));
    }

    // PLACEHOLDER: divergence check
    // double divergence = compute_divergence(u, v, w, dims.nx, dims.ny, dims.nz, dims.dx(), dims.dy(), dims.dz());
    // ASSERT_NEAR(divergence, 0.0, 1e-6);
}

} // namespace navier_stokes_solver
