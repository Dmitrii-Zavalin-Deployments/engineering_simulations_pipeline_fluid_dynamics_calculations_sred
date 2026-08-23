/**
 * @file test_mass_continuity.cpp
 * @brief Literate integration test verifying mass continuity (div(u) = 0) 
 *        and numerical divergence constraints in 3D incompressible Navier-Stokes flow.
 *
 * --- LITERATE TEST NARRATIVE ---
 * 
 * 1. Setup & Initialization:
 *    We configure a uniform 10x10x10 Cartesian grid with equal spacing (dx = dy = dz = 0.1) 
 *    and establish solid boundary walls (mask = -1) along all outer domain boundaries, 
 *    with interior cells designated as active fluid (mask = 1).
 *
 * 2. Time-Step Advancement:
 *    We invoke the NavierStokesOrchestrator to advance the velocity and pressure fields 
 *    by a single time-step (dt = 0.001), executing advection, diffusion, and the pressure 
 *    Poisson projection step required to enforce mass conservation.
 *
 * 3. Divergence & Continuity Validation:
 *    We compute discrete central-difference spatial derivatives across all interior 
 *    fluid cells to evaluate velocity divergence:
 *        div(u) = du/dx + dv/dy + dw/dz
 *    We then assert that local maximum divergence and global mean volume flux 
 *    satisfy strict numerical convergence criteria.
 */

#include <gtest/gtest.h>
#include "orchestrator.hpp"
#include "grid_math.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <iostream>

namespace navier_stokes_solver {

/**
 * @class MassContinuityIntegrationTest
 * @brief Establishes the 3D computational domain, boundary walls, and initial velocity 
 *        fields for evaluating mass conservation following the pressure projection step.
 */
class MassContinuityIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // We configure a uniform 10x10x10 cartesian grid with equal spacing 
        // dx = dy = dz = 0.1 across all spatial dimensions.
        dims_ = {10, 10, 10, 0.1, 0.1, 0.1};
        
        // The solver configuration sets the maximum iterative solver iterations, 
        // convergence tolerance, and reference fluid density (rho = 1.0).
        config_ = {5000, 1e-8, 1.0};

        total_cells_ = static_cast<size_t>(dims_.nx) * dims_.ny * dims_.nz;

        // Allocate velocity components, pressure field, and external body forces.
        u_.resize(total_cells_, 0.0);
        v_.resize(total_cells_, 0.0);
        w_.resize(total_cells_, 0.0);
        p_.resize(total_cells_, 0.0);

        fx_.resize(total_cells_, 0.0);
        fy_.resize(total_cells_, 0.0);
        fz_.resize(total_cells_, 0.0);
        
        gravity_ = {0.0, 0.0, 0.0};
        
        // Initialize the domain cell mask, setting interior cells as active fluid (1).
        mask_.resize(total_cells_, 1); 

        // We establish solid boundary walls (mask == -1) along all outer domain faces.
        for (int k = 0; k < dims_.nz; ++k) {
            for (int j = 0; j < dims_.ny; ++j) {
                for (int i = 0; i < dims_.nx; ++i) {
                    if (i == 0 || i == dims_.nx - 1 || 
                        j == 0 || j == dims_.ny - 1 || 
                        k == 0 || k == dims_.nz - 1) {
                        size_t idx = static_cast<size_t>(get_flat_index(i, j, k, dims_.nx, dims_.ny));
                        mask_[idx] = -1;
                    }
                }
            }
        }

        // Apply standard boundary conditions, including a lid-driven top wall 
        // moving in the positive X-direction with velocity u = 1.0.
        bc_list_ = {
            {"wall", "no-slip", 0.0, 0.0, 0.0, 0.0},
            {"z_max", "no-slip", 1.0, 0.0, 0.0, 0.0}
        };
    }

    GridDimensions dims_;
    SolverConfig config_;
    size_t total_cells_;

    std::vector<double> u_, v_, w_, p_;
    std::vector<double> fx_, fy_, fz_;
    std::vector<double> gravity_;
    std::vector<int> mask_;
    std::vector<BoundaryCondition> bc_list_;
};

/**
 * @test EnforcesZeroDivergenceInFluidDomain
 * @brief Advances the flow field by one time-step via the NavierStokesOrchestrator 
 *        and verifies that the resulting velocity field satisfies the incompressible 
 *        continuity constraint (div(u) = 0).
 */
TEST_F(MassContinuityIntegrationTest, EnforcesZeroDivergenceInFluidDomain) {
    std::cout << "[debug] MassContinuityIntegrationTest.EnforcesZeroDivergenceInFluidDomain starting\n";

    NavierStokesOrchestrator orchestrator(dims_, config_);

    const double dt = 0.001;
    const double mu = 0.01;

    std::cout << "[debug] Calling orchestrator.step(dt=" << dt << ", mu=" << mu << ")\n";

    // We execute a single solver step to apply advection, diffusion, 
    // and the pressure Poisson projection necessary for enforcing mass conservation.
    orchestrator.step(dt, mu, gravity_, fx_, fy_, fz_, mask_, bc_list_, u_, v_, w_, p_);

    std::cout << "[debug] Finished orchestrator.step()\n";

    double max_divergence = 0.0;
    double total_divergence = 0.0;
    int interior_fluid_count = 0;

    // We compute the discrete velocity divergence across the interior fluid domain using 
    // central differences: div(u) = du/dx + dv/dy + dw/dz
    for (int k = 1; k < dims_.nz - 1; ++k) {
        for (int j = 1; j < dims_.ny - 1; ++j) {
            for (int i = 1; i < dims_.nx - 1; ++i) {
                size_t idx = static_cast<size_t>(get_flat_index(i, j, k, dims_.nx, dims_.ny));

                if (mask_[idx] == 1) {
                    size_t idx_e = static_cast<size_t>(get_flat_index(i + 1, j, k, dims_.nx, dims_.ny));
                    size_t idx_w = static_cast<size_t>(get_flat_index(i - 1, j, k, dims_.nx, dims_.ny));
                    size_t idx_n = static_cast<size_t>(get_flat_index(i, j + 1, k, dims_.nx, dims_.ny));
                    size_t idx_s = static_cast<size_t>(get_flat_index(i, j - 1, k, dims_.nx, dims_.ny));
                    size_t idx_t = static_cast<size_t>(get_flat_index(i, j, k + 1, dims_.nx, dims_.ny));
                    size_t idx_b = static_cast<size_t>(get_flat_index(i, j, k - 1, dims_.nx, dims_.ny));

                    double dudx = (u_[idx_e] - u_[idx_w]) / (2.0 * dims_.dx);
                    double dvdy = (v_[idx_n] - v_[idx_s]) / (2.0 * dims_.dy);
                    double dwdz = (w_[idx_t] - w_[idx_b]) / (2.0 * dims_.dz);

                    double div_u = dudx + dvdy + dwdz;

                    max_divergence = std::max(max_divergence, std::abs(div_u));
                    total_divergence += div_u;
                    interior_fluid_count++;
                }
            }
        }
    }

    std::cout << "[debug] interior_fluid_count=" << interior_fluid_count << "\n";

    // Ensure that valid interior fluid cells were successfully evaluated.
    ASSERT_GT(interior_fluid_count, 0) 
        << "Error: No active fluid cells found in integration test domain.";

    double mean_divergence = total_divergence / static_cast<double>(interior_fluid_count);

    std::cout << "[debug] max_divergence=" << max_divergence 
              << " mean_divergence=" << mean_divergence << "\n";

    // Assertion 1: Local divergence must remain below the strict numerical tolerance threshold.
    ASSERT_LT(max_divergence, 2e-2) 
        << "Local mass conservation failure: Maximum velocity divergence exceeds physical tolerance.";
    
    // Assertion 2: Global mean divergence across the domain must evaluate near zero.
    ASSERT_NEAR(mean_divergence, 0.0, 1e-5) 
        << "Global mass conservation failure: Net domain volume flux is non-zero.";

    std::cout << "[debug] MassContinuityIntegrationTest.EnforcesZeroDivergenceInFluidDomain completed successfully\n";
}

} // namespace navier_stokes_solver
