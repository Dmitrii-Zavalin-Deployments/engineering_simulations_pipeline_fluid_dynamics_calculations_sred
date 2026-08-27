/**
 * @file test_full_pipeline_literate.cpp
 * @brief Literate-style integration test for the full Navier–Stokes solver pipeline.
 *
 * This test evaluates the complete end-to-end execution of the solver pipeline.
 * Rather than invoking individual modules manually, it delegates execution entirely
 * to NavierStokesOrchestrator::step() and inspects intermediate state snapshots
 * captured after each stage.
 */

#ifndef NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
#define NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
#endif

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>

#include "orchestrator.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"
#include "ghost_handler.hpp"

namespace navier_stokes_solver {

TEST(FullPipelineLiterateTest, StepByStepMicroManaged) {

    // ============================================================================
    // SECTION 1 — Grid Setup
    // ============================================================================

    GridDimensions dims;

    double x_min = 0.0;
    double x_max = 4.0;
    double y_min = 0.0;
    double y_max = 4.0;
    double z_min = 0.0;
    double z_max = 2.0;

    dims.nx = 8;
    dims.ny = 8;
    dims.nz = 4;

    dims.dx = (x_max - x_min) / dims.nx;
    dims.dy = (y_max - y_min) / dims.ny;
    dims.dz = (z_max - z_min) / dims.nz;

    dims.validate();

    const size_t total_cells = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;

    // ============================================================================
    // SECTION 2 — Allocate Fields
    // ============================================================================

    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

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
    std::vector<double> fx(total_cells, 0.0);
    std::vector<double> fy(total_cells, 0.0);
    std::vector<double> fz(total_cells, 0.0);

    // ============================================================================
    // SECTION 5 — Execute Pipeline via Orchestrator
    // ============================================================================

    NavierStokesOrchestrator orchestrator(dims, config);

    // Execute the full integrated pipeline
    orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p);

    // Retrieve debug snapshots generated by orchestrator.step()
    const auto& snapshots = orchestrator.get_debug_snapshots();
    ASSERT_FALSE(snapshots.empty());

    // Helper lambda to fetch a snapshot by key safely
    auto get_snapshot = [&](const std::string& stage_name) -> const PipelineSnapshot& {
        auto it = snapshots.find(stage_name);
        EXPECT_NE(it, snapshots.end()) << "Missing snapshot for stage: " << stage_name;
        return it->second;
    };

    // ============================================================================
    // SECTION 6 — Verify Stage 1 Snapshot: Pre-Step
    // ============================================================================

    {
        const auto& snap = get_snapshot("pre_step");

        for (int k = 0; k < dims.nz; ++k) {
            for (int j = 0; j < dims.ny; ++j) {
                for (int i = 0; i < dims.nx; ++i) {
                    const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, dims.nx, dims.ny));

                    // 1. Wall cells (mask == -1) clamped to zero
                    if (mask[idx] == -1) {
                        ASSERT_NEAR(snap.u[idx], 0.0, 1e-12);
                        ASSERT_NEAR(snap.v[idx], 0.0, 1e-12);
                        ASSERT_NEAR(snap.w[idx], 0.0, 1e-12);
                    }

                    // 2. Solid interior cells (mask == 0) clamped to zero
                    if (mask[idx] == 0) {
                        ASSERT_NEAR(snap.u[idx], 0.0, 1e-12);
                        ASSERT_NEAR(snap.v[idx], 0.0, 1e-12);
                        ASSERT_NEAR(snap.w[idx], 0.0, 1e-12);
                    }

                    // 3. Inflow plane (z_min)
                    if (k == 0 && mask[idx] == 1) {
                        ASSERT_NEAR(snap.w[idx], 1.0, 1e-12);
                    }

                    // 4. Outflow plane (z_max)
                    if (k == dims.nz - 1 && mask[idx] == 1) {
                        ASSERT_NEAR(snap.w[idx], 1.0, 1e-12);
                    }

                    // 5. Finite check
                    ASSERT_TRUE(std::isfinite(snap.u[idx]));
                    ASSERT_TRUE(std::isfinite(snap.v[idx]));
                    ASSERT_TRUE(std::isfinite(snap.w[idx]));
                    ASSERT_TRUE(std::isfinite(snap.p[idx]));
                }
            }
        }
    }

    // ============================================================================
    // SECTION 7 — Verify Stage 2 Snapshot: Predictor
    // ============================================================================

    {
        const auto& snap = get_snapshot("predictor");
        const auto& pre_snap = get_snapshot("pre_step");

        for (int k = 0; k < dims.nz; ++k) {
            for (int j = 0; j < dims.ny; ++j) {
                for (int i = 0; i < dims.nx; ++i) {
                    const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, dims.nx, dims.ny));

                    // 1. Finite checks on trial velocities
                    ASSERT_TRUE(std::isfinite(snap.u_star[idx]));
                    ASSERT_TRUE(std::isfinite(snap.v_star[idx]));
                    ASSERT_TRUE(std::isfinite(snap.w_star[idx]));

                    // 2. Solid/wall cells must preserve pre-step values
                    if (mask[idx] != 1) {
                        ASSERT_NEAR(snap.u_star[idx], pre_snap.u[idx], 1e-12);
                        ASSERT_NEAR(snap.v_star[idx], pre_snap.v[idx], 1e-12);
                        ASSERT_NEAR(snap.w_star[idx], pre_snap.w[idx], 1e-12);
                    }
                }
            }
        }
    }

    // ============================================================================
    // SECTION 8 — Verify Stage 3 Snapshot: Poisson Solver
    // ============================================================================

    {
        const auto& snap = get_snapshot("poisson");

        for (size_t idx = 0; idx < total_cells; ++idx) {
            ASSERT_TRUE(std::isfinite(snap.p[idx]));
        }
    }

    // ============================================================================
    // SECTION 9 — Verify Stage 4 Snapshot: Corrector
    // ============================================================================

    {
        const auto& snap = get_snapshot("corrector");

        for (size_t idx = 0; idx < total_cells; ++idx) {
            ASSERT_TRUE(std::isfinite(snap.u[idx]));
            ASSERT_TRUE(std::isfinite(snap.v[idx]));
            ASSERT_TRUE(std::isfinite(snap.w[idx]));

            // Solid and wall boundaries remain non-penetrating / no-slip
            if (mask[idx] != 1) {
                ASSERT_NEAR(snap.u[idx], 0.0, 1e-12);
                ASSERT_NEAR(snap.v[idx], 0.0, 1e-12);
                ASSERT_NEAR(snap.w[idx], 0.0, 1e-12);
            }
        }
    }

    // ============================================================================
    // SECTION 10 — Verify Stage 5 Snapshot: Ghost Sync
    // ============================================================================

    {
        const auto& snap = get_snapshot("ghost_sync_2");

        for (size_t idx = 0; idx < total_cells; ++idx) {
            ASSERT_TRUE(std::isfinite(snap.u[idx]));
            ASSERT_TRUE(std::isfinite(snap.v[idx]));
            ASSERT_TRUE(std::isfinite(snap.w[idx]));
            ASSERT_TRUE(std::isfinite(snap.p[idx]));
        }
    }

    // ============================================================================
    // SECTION 11 — Final Output Verification
    // ============================================================================

    for (size_t idx = 0; idx < total_cells; ++idx) {
        ASSERT_TRUE(std::isfinite(u[idx]));
        ASSERT_TRUE(std::isfinite(v[idx]));
        ASSERT_TRUE(std::isfinite(w[idx]));
        ASSERT_TRUE(std::isfinite(p[idx]));

        if (mask[idx] != 1) {
            ASSERT_NEAR(u[idx], 0.0, 1e-12);
            ASSERT_NEAR(v[idx], 0.0, 1e-12);
            ASSERT_NEAR(w[idx], 0.0, 1e-12);
        }
    }

    // ============================================================================
    // SECTION 12 — Verify Fluid Core Streamwise Uniformity (u=0, v=0, w=1)
    // ============================================================================

    /**
     * In a straight duct with uniform inlet/outlet (w = 1.0):
     *   - Transverse components (u, v) must remain zero across the fluid core.
     *   - Streamwise velocity (w) must propagate through all internal fluid layers (k=1, 2).
     */
    for (int k = 0; k < dims.nz; ++k) {
        for (int j = 0; j < dims.ny; ++j) {
            for (int i = 0; i < dims.nx; ++i) {
                const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, dims.nx, dims.ny));

                if (mask[idx] == 1) {
                    // No transverse flow in a straight uniform channel
                    ASSERT_NEAR(u[idx], 0.0, 1e-6) 
                        << "Non-zero u velocity at fluid cell (" << i << ", " << j << ", " << k << ")";
                    ASSERT_NEAR(v[idx], 0.0, 1e-6) 
                        << "Non-zero v velocity at fluid cell (" << i << ", " << j << ", " << k << ")";

                    // Streamwise flow must propagate through the entire core
                    ASSERT_NEAR(w[idx], 1.0, 1e-2) 
                        << "Inconsistent streamwise w velocity at fluid cell (" << i << ", " << j << ", " << k << ")";
                }
            }
        }
    }
}

} // namespace navier_stokes_solver