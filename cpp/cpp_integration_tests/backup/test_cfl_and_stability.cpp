/**
 * @file test_cfl_and_stability.cpp
 * @brief Integration test verifying Courant-Friedrichs-Lewy (CFL) condition enforcement,
 *        temporal stability bounds, and safety intercept exceptions under numerical velocity spikes.
 *
 * LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
 * ---------------------------------------------------------------------------------
 * Temporal stability in explicit and semi-implicit advection solvers is governed by 
 * the Courant-Friedrichs-Lewy (CFL) condition. For a 3D Eulerian grid, the dimensionless 
 * CFL number C measures the distance information travels across grid cells during a time step dt:
 * 
 *     C = max( (|u|_max * dt) / dx, (|v|_max * dt) / dy, (|w|_max * dt) / dz ) <= C_max
 * 
 * Where C_max = 1.0 represents the hyperbolic stability boundary (information cannot 
 * traverse more than one mesh cell per discrete time step).
 * 
 * TEST SCENARIOS:
 *   - Scenario 6.1 (Case A - Stable):
 *       dx = 0.01 m, u_max = 10.0 m/s, dt = 0.0005 s ==> C = (10.0 * 0.0005) / 0.01 = 0.5 <= 1.0
 *       Expectation: Execution completes cleanly, velocity field remains finite, 
 *                    and numerical divergence is successfully suppressed/bounded by projection.
 * 
 *   - Scenario 6.1 (Case B - CFL Violation):
 *       dx = 0.01 m, u_max = 10.0 m/s, dt = 0.002 s ==> C = (10.0 * 0.002) / 0.01 = 2.0 > 1.0
 *       Expectation: The Orchestrator's CFL guard intercepts the time-step update and 
 *                    throws an exception or prevents numeric NaN divergence blow-up.
 * ---------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include "orchestrator.hpp"
#include "grid_math.hpp"
#include "boundary_condition.hpp"

using namespace navier_stokes_solver;

/**
 * @class CflAndStabilityTest
 * @brief Test fixture for setting up uniform domain metrics and velocity initialization fields
 *        written following a literate narrative style.
 */
class CflAndStabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // We define the spatial grid spacing uniformly across all three dimensions:
        //     dx = dy = dz = 0.01 m
        nx_ = 10;
        ny_ = 10;
        nz_ = 10;
        dx_ = 0.01;
        dy_ = 0.01;
        dz_ = 0.01;

        dims_ = GridDimensions{nx_, ny_, nz_, dx_, dy_, dz_};
        total_cells_ = static_cast<size_t>(nx_) * ny_ * nz_;

        density_ = 1.0;
        mu_ = 0.001;

        config_.density = density_;
        config_.max_poisson_iterations = 1000;
        config_.poisson_tolerance = 1e-6;

        // We initialize mask and external body force arrays for the fluid domain.
        mask_ = std::vector<int>(total_cells_, 1); 
        fx_ = std::vector<double>(total_cells_, 0.0);
        fy_ = std::vector<double>(total_cells_, 0.0);
        fz_ = std::vector<double>(total_cells_, 0.0);
        gravity_ = {0.0, 0.0, 0.0};

        BoundaryCondition bc_wall;
        bc_wall.location = "wall";
        bc_wall.type = "no-slip";
        bc_wall.u_val = 0.0;
        bc_wall.v_val = 0.0;
        bc_wall.w_val = 0.0;
        bc_list_.push_back(bc_wall);
    }

    /**
     * @brief Computes maximum interior velocity divergence using central differences:
     *     div(u) = du/dx + dv/dy + dw/dz = 0
     */
    double ComputeMaxDivergence(
        const std::vector<double>& u,
        const std::vector<double>& v,
        const std::vector<double>& w
    ) const {
        double max_div = 0.0;
        for (int k = 1; k < nz_ - 1; ++k) {
            for (int j = 1; j < ny_ - 1; ++j) {
                for (int i = 1; i < nx_ - 1; ++i) {
                    size_t idx_px = (i + 1) + static_cast<size_t>(nx_) * (j + ny_ * k);
                    size_t idx_nx = (i - 1) + static_cast<size_t>(nx_) * (j + ny_ * k);
                    size_t idx_py = i + static_cast<size_t>(nx_) * ((j + 1) + ny_ * k);
                    size_t idx_ny = i + static_cast<size_t>(nx_) * ((j - 1) + ny_ * k);
                    size_t idx_pz = i + static_cast<size_t>(nx_) * (j + ny_ * (k + 1));
                    size_t idx_nz = i + static_cast<size_t>(nx_) * (j + ny_ * (k - 1));

                    double du_dx = (u[idx_px] - u[idx_nx]) / (2.0 * dx_);
                    double dv_dy = (v[idx_py] - v[idx_ny]) / (2.0 * dy_);
                    double dw_dz = (w[idx_pz] - w[idx_nz]) / (2.0 * dz_);

                    double div = std::abs(du_dx + dv_dy + dw_dz);
                    max_div = std::max(max_div, div);
                }
            }
        }
        return max_div;
    }

    int nx_, ny_, nz_;
    double dx_, dy_, dz_;
    GridDimensions dims_;
    size_t total_cells_;
    double density_;
    double mu_;
    SolverConfig config_;

    std::vector<int> mask_;
    std::vector<double> fx_, fy_, fz_, gravity_;
    std::vector<BoundaryCondition> bc_list_;
};

// =================================================================================
// Scenario 6.1: CFL Violation Safety Intercept & Stability Bounds
// =================================================================================
TEST_F(CflAndStabilityTest, CflViolationSafetyIntercept) {
    NavierStokesOrchestrator orchestrator(dims_, config_);

    // We set up base velocity and pressure fields.
    std::vector<double> u_base(total_cells_, 0.0);
    std::vector<double> v_base(total_cells_, 0.0);
    std::vector<double> w_base(total_cells_, 0.0);
    std::vector<double> p_base(total_cells_, 0.0);

    // We inject a local velocity spike u_max = 10.0 m/s at domain center cell (i=5, j=5, k=5).
    size_t center_idx = 5 + static_cast<size_t>(nx_) * (5 + ny_ * 5);
    u_base[center_idx] = 10.0;

    // -----------------------------------------------------------------------------
    // Case A (Stable): dt = 0.0005 s ==> Courant number C = (10.0 * 0.0005) / 0.01 = 0.5 <= 1.0
    // -----------------------------------------------------------------------------
    {
        double dt_stable = 0.0005;
        std::vector<double> u = u_base;
        std::vector<double> v = v_base;
        std::vector<double> w = w_base;
        std::vector<double> p = p_base;

        // We compute initial divergence prior to step execution:
        double initial_div = ComputeMaxDivergence(u, v, w);
        assert(initial_div >= 0.0);

        // Step execution must complete without throwing exceptions under stable CFL conditions.
        assert(dt_stable > 0.0);
        ASSERT_NO_THROW({
            orchestrator.step(dt_stable, mu_, gravity_, fx_, fy_, fz_, mask_, bc_list_, u, v, w, p);
        }) << "Case A Failed: Orchestrator threw an unexpected exception under stable CFL conditions (C = 0.5).";

        // We verify that no NaN or Infinite values appear in the output velocity field.
        for (size_t i = 0; i < total_cells_; ++i) {
            ASSERT_FALSE(std::isnan(u[i]) || std::isinf(u[i])) << "Case A Failed: u contains NaN/Inf at index " << i;
            ASSERT_FALSE(std::isnan(v[i]) || std::isinf(v[i])) << "Case A Failed: v contains NaN/Inf at index " << i;
            ASSERT_FALSE(std::isnan(w[i]) || std::isinf(w[i])) << "Case A Failed: w contains NaN/Inf at index " << i;
        }

        // We verify the stability invariant: pressure projection must suppress or bound the divergence spike.
        double final_div = ComputeMaxDivergence(u, v, w);
        assert(final_div >= 0.0);
        ASSERT_LE(final_div, initial_div) 
            << "Case A Failed: Divergence grew from " << initial_div << " to " << final_div 
            << " under stable CFL conditions, violating solver stability bounds.";
    }

    // -----------------------------------------------------------------------------
    // Case B (Unstable): dt = 0.002 s ==> Courant number C = (10.0 * 0.002) / 0.01 = 2.0 > 1.0
    // -----------------------------------------------------------------------------
    {
        double dt_unstable = 0.002;
        std::vector<double> u = u_base;
        std::vector<double> v = v_base;
        std::vector<double> w = w_base;
        std::vector<double> p = p_base;

        // We verify that the orchestrator either throws an exception or catches/guards the instability.
        bool guard_triggered = false;
        try {
            orchestrator.step(dt_unstable, mu_, gravity_, fx_, fy_, fz_, mask_, bc_list_, u, v, w, p);
            
            // If no exception was thrown, verify whether numerical safety guards prevented NaN explosion.
            for (size_t i = 0; i < total_cells_; ++i) {
                if (std::isnan(u[i]) || std::isinf(u[i]) || std::abs(u[i]) > 1000.0) {
                    guard_triggered = false;
                    break;
                }
                guard_triggered = true;
            }
        } catch (const std::exception& e) {
            guard_triggered = true;
        }

        ASSERT_TRUE(guard_triggered) 
            << "Case B Failed: Orchestrator failed to guard against CFL violation (C = 2.0 > 1.0) "
            << "and allowed unhandled numerical instability or NaN explosion.";
    }
}
