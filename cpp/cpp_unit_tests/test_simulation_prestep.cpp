/**
 * @file test_simulation_prestep.cpp
 * @brief Literate Test Suite for C++ Pre-Step Boundary & Initial Condition Setup.
 *
 * This test suite acts as a literate narrative document for verifying the pre-step boundary
 * condition and spatial initialization subsystem (simulation_prestep.cpp), including the
 * strict non-overwriting exclusivity policy and fallback wall assignment rules.
 * Explanatory text and physical principles are documented in prose comments, while
 * mathematical constraints and boundary assertions are executed via Google Test assertions.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "simulation_prestep.hpp"
#include "orchestrator.hpp"
#include "grid_math.hpp"

namespace navier_stokes_solver {

// ============================================================================
// NARRATIVE SECTION 1: Grid Dimension and Contract Safety Validation
// ============================================================================
// The pre-step routine enforces strict geometric and memory safety contracts:
//
//      1. Geometric bounds:
//         nx >= 3  AND  ny >= 3  AND  nz >= 3
//
//      2. Buffer sizing equality:
//         total_cells = nx * ny * nz
//         size(u) == size(v) == size(w) == size(p) == size(mask) == total_cells
//
// Violations of these constraints throw std::invalid_argument exceptions,
// explicitly exercising lines 43 and 49 in simulation_prestep.cpp.
// ============================================================================

TEST(SimulationPrestepTest, InvalidGridDimensions) {
    // A 3D fluid grid requires at least 3 cells along each spatial axis (nx >= 3, ny >= 3, nz >= 3)
    // to establish distinct interior and boundary domain interfaces.
    //
    // We evaluate invalid grid configurations where each axis dimension falls below 3:
    //      Case A: nx = 2, ny = 4, nz = 4  (nx < 3)
    //      Case B: nx = 4, ny = 2, nz = 4  (ny < 3)
    //      Case C: nx = 4, ny = 4, nz = 2  (nz < 3)
    std::vector<BoundaryCondition> bc_list;

    // Case A: Invalid nx dimension (nx = 2)
    {
        int nx = 2, ny = 4, nz = 4;
        size_t total_cells = static_cast<size_t>(nx) * ny * nz;
        std::vector<double> u(total_cells, 0.0), v(total_cells, 0.0), w(total_cells, 0.0), p(total_cells, 0.0);
        std::vector<int> mask(total_cells, 1);

        // Line 43 execution: Grid dimensions smaller than 3x3x3 throw GEOMETRY ERROR
        EXPECT_THROW(
            execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz),
            std::invalid_argument
        );
    }

    // Case B: Invalid ny dimension (ny = 2)
    {
        int nx = 4, ny = 2, nz = 4;
        size_t total_cells = static_cast<size_t>(nx) * ny * nz;
        std::vector<double> u(total_cells, 0.0), v(total_cells, 0.0), w(total_cells, 0.0), p(total_cells, 0.0);
        std::vector<int> mask(total_cells, 1);

        EXPECT_THROW(
            execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz),
            std::invalid_argument
        );
    }

    // Case C: Invalid nz dimension (nz = 2)
    {
        int nx = 4, ny = 4, nz = 2;
        size_t total_cells = static_cast<size_t>(nx) * ny * nz;
        std::vector<double> u(total_cells, 0.0), v(total_cells, 0.0), w(total_cells, 0.0), p(total_cells, 0.0);
        std::vector<int> mask(total_cells, 1);

        EXPECT_THROW(
            execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz),
            std::invalid_argument
        );
    }
}

TEST(SimulationPrestepTest, FieldVectorSizeMismatch) {
    // Memory safety requires every state tensor buffer (u, v, w, p, mask) to exactly match:
    //      total_cells = nx * ny * nz
    //
    // Any buffer size mismatch triggers line 49 in simulation_prestep.cpp.
    int nx = 4, ny = 4, nz = 4;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    std::vector<BoundaryCondition> bc_list;

    // Test size mismatch in velocity component u buffer (size = total_cells - 1)
    {
        std::vector<double> u(total_cells - 1, 0.0), v(total_cells, 0.0), w(total_cells, 0.0), p(total_cells, 0.0);
        std::vector<int> mask(total_cells, 1);

        // Line 49 execution: Buffer size mismatch throws CONTRACT VIOLATION
        EXPECT_THROW(
            execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz),
            std::invalid_argument
        );
    }

    // Test size mismatch in mask buffer (size = total_cells + 5)
    {
        std::vector<double> u(total_cells, 0.0), v(total_cells, 0.0), w(total_cells, 0.0), p(total_cells, 0.0);
        std::vector<int> mask(total_cells + 5, 1);

        EXPECT_THROW(
            execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz),
            std::invalid_argument
        );
    }
}

// ============================================================================
// NARRATIVE SECTION 2: Boundary Condition Dispatch and State Updates
// ============================================================================
// Pre-step boundary application prescribes Dirichlet values or Neumann gradients:
//
//      1. Inflow Dirichlet condition:
//         u[idx] = u_val,  v[idx] = v_val,  w[idx] = w_val
//
//      2. Pressure & Outflow Neumann condition:
//         p[idx] = scalar_p
//         du/dn = 0  =>  u[boundary] = u[adjacent_interior]
//
//      3. No-slip Dirichlet condition:
//         u[idx] = 0.0,  v[idx] = 0.0,  w[idx] = 0.0
//
//      4. Free-slip condition:
//         u_normal = 0.0,  d(u_tangential)/dn = 0
// ============================================================================

TEST(SimulationPrestepTest, BoundaryConditionDispatch) {
    int nx = 4, ny = 4, nz = 4;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> u(total_cells, 1.0);
    std::vector<double> v(total_cells, 1.0);
    std::vector<double> w(total_cells, 1.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);

    // Construct boundary condition list covering all schema types and domain locations
    BoundaryCondition bc_inflow;
    bc_inflow.location = "x_min";
    bc_inflow.type = "inflow";
    bc_inflow.u_val = 5.0;
    bc_inflow.v_val = 0.0;
    bc_inflow.w_val = 0.0;

    BoundaryCondition bc_outflow;
    bc_outflow.location = "x_max";
    bc_outflow.type = "outflow";

    BoundaryCondition bc_noslip;
    bc_noslip.location = "y_min";
    bc_noslip.type = "no-slip";

    BoundaryCondition bc_freeslip;
    bc_freeslip.location = "y_max";
    bc_freeslip.type = "free-slip";

    BoundaryCondition bc_pressure;
    bc_pressure.location = "z_max";
    bc_pressure.type = "pressure";
    bc_pressure.scalar_p = 101325.0;

    BoundaryCondition bc_wall_pressure;
    bc_wall_pressure.location = "wall";
    bc_wall_pressure.type = "pressure";

    BoundaryCondition bc_wall_freeslip;
    bc_wall_freeslip.location = "wall";
    bc_wall_freeslip.type = "free-slip";

    std::vector<BoundaryCondition> bc_list = {
        bc_inflow, bc_outflow, bc_noslip, bc_freeslip,
        bc_pressure, bc_wall_pressure, bc_wall_freeslip
    };

    // Execute pre-step boundary setup routine
    EXPECT_NO_THROW(execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz));

    // Verify all buffer values remain finite and initialized
    for (size_t i = 0; i < total_cells; ++i) {
        EXPECT_TRUE(std::isfinite(u[i]));
        EXPECT_TRUE(std::isfinite(v[i]));
        EXPECT_TRUE(std::isfinite(w[i]));
        EXPECT_TRUE(std::isfinite(p[i]));
    }
}

// ============================================================================
// NARRATIVE SECTION 3: Non-Overwriting Exclusivity and Wall Shorthand Verification
// ============================================================================
// The solver enforces a strict non-overwriting policy where explicit face-specific
// boundary rules (e.g. "x_min", "y_max") take absolute precedence, and generic
// catch-all "wall" rules only apply to unclaimed boundary cells.
// ============================================================================

TEST(SimulationPrestepTest, WallDoesNotOverrideExplicitFaces) {
    // Verify that a generic catch-all "wall" rule does not overwrite or clobber
    // an explicit face-specific boundary rule (e.g. an inflow condition on x_min).
    int nx = 4, ny = 4, nz = 4;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> u(total_cells, 1.0);
    std::vector<double> v(total_cells, 1.0);
    std::vector<double> w(total_cells, 1.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);

    // Rule 1: Explicit inflow on x_min with distinctive velocity values (u = 7.5)
    BoundaryCondition bc_explicit_inflow;
    bc_explicit_inflow.location = "x_min";
    bc_explicit_inflow.type = "inflow";
    bc_explicit_inflow.u_val = 7.5;
    bc_explicit_inflow.v_val = 2.0;
    bc_explicit_inflow.w_val = 3.0;

    // Rule 2: Generic wall rule with no-slip (u = 0.0) placed after the explicit rule
    BoundaryCondition bc_generic_wall;
    bc_generic_wall.location = "wall";
    bc_generic_wall.type = "no-slip";

    // Even though the generic wall rule specifies "no-slip", the explicit x_min cells
    // must remain claimed and protected by Pass 1.
    std::vector<BoundaryCondition> bc_list = {bc_explicit_inflow, bc_generic_wall};

    EXPECT_NO_THROW(execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz));

    // Verify cells on x_min (i = 0) retain the explicit inflow values (7.5, 2.0, 3.0)
    // rather than being overwritten to 0.0 by the generic wall no-slip rule.
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            size_t idx = static_cast<size_t>(get_flat_index(0, j, k, nx, ny));
            EXPECT_DOUBLE_EQ(u[idx], 7.5);
            EXPECT_DOUBLE_EQ(v[idx], 2.0);
            EXPECT_DOUBLE_EQ(w[idx], 3.0);
        }
    }
}

TEST(SimulationPrestepTest, WallLocationAssignmentCorrectness) {
    // Verify that the "wall" location correctly assigns boundary conditions to all
    // outer boundary cells (where i=0, i=nx-1, j=0, j=ny-1, k=0, or k=nz-1)
    // when no explicit face rules override them.
    int nx = 4, ny = 4, nz = 4;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> u(total_cells, 1.0);
    std::vector<double> v(total_cells, 1.0);
    std::vector<double> w(total_cells, 1.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);

    // Apply a pure no-slip wall condition across all boundary surfaces using the "wall" shorthand
    BoundaryCondition bc_wall;
    bc_wall.location = "wall";
    bc_wall.type = "no-slip";

    std::vector<BoundaryCondition> bc_list = {bc_wall};

    EXPECT_NO_THROW(execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz));

    // Verify boundary cells have velocity set to 0.0 (no-slip)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
                    size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    EXPECT_DOUBLE_EQ(u[idx], 0.0);
                    EXPECT_DOUBLE_EQ(v[idx], 0.0);
                    EXPECT_DOUBLE_EQ(w[idx], 0.0);
                }
            }
        }
    }
}

} // namespace navier_stokes_solver
