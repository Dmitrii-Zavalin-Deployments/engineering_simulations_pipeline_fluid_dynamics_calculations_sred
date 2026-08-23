#!/bin/bash
set -euo pipefail

TARGET_FILE="cpp/cpp_integration_tests/test_mass_continuity.cpp"

echo "📌 1. Rewriting $TARGET_FILE with clean divergence loop..."

cat << 'CPP_EOF' > "$TARGET_FILE"
/**
 * @file test_mass_continuity.cpp
 * @brief Literate integration test verifying mass continuity (div(u) = 0) 
 *        and numerical divergence constraints in 3D incompressible Navier-Stokes flow.
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

class MassContinuityIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        dims_ = {10, 10, 10, 0.1, 0.1, 0.1};
        config_ = {5000, 1e-8, 1.0};

        total_cells_ = static_cast<size_t>(dims_.nx) * dims_.ny * dims_.nz;

        u_.resize(total_cells_, 0.0);
        v_.resize(total_cells_, 0.0);
        w_.resize(total_cells_, 0.0);
        p_.resize(total_cells_, 0.0);

        fx_.resize(total_cells_, 0.0);
        fy_.resize(total_cells_, 0.0);
        fz_.resize(total_cells_, 0.0);
        
        gravity_ = {0.0, 0.0, 0.0};
        mask_.resize(total_cells_, 1); 

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

TEST_F(MassContinuityIntegrationTest, EnforcesZeroDivergenceInFluidDomain) {
    std::cout << "[debug] MassContinuityIntegrationTest starting\n";

    NavierStokesOrchestrator orchestrator(dims_, config_);

    const double dt = 0.001;
    const double mu = 0.01;

    orchestrator.step(dt, mu, gravity_, fx_, fy_, fz_, mask_, bc_list_, u_, v_, w_, p_);

    double max_divergence = 0.0;
    double total_divergence = 0.0;
    int interior_fluid_count = 0;

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
                    total_divergence += std::abs(div_u);
                    interior_fluid_count++;
                }
            }
        }
    }

    double mean_divergence = (interior_fluid_count > 0) ? (total_divergence / interior_fluid_count) : 0.0;
    std::cout << "[debug] max_divergence=" << max_divergence << " mean_divergence=" << mean_divergence << "\n";

    EXPECT_LT(max_divergence, 1.0);
    EXPECT_LT(mean_divergence, 0.5);
}

} // namespace navier_stokes_solver
CPP_EOF

echo "📌 2. Rebuilding and running test_mass_continuity..."
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target test_mass_continuity -j$(nproc)
./build/cpp/cpp_integration_tests/test_mass_continuity