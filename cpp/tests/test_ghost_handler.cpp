/**
 * @file test_ghost_handler.cpp
 * @brief Literate test suite for Ghost Trial Buffer Synchronization.
 * 
 * This test file narrates and verifies the memory alignment copy accuracy 
 * and null-pointer contract violation handling of the C++ sync_ghost_trial_buffers 
 * cleanup routine.
 */

#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include "ghost_handler.hpp"

using namespace ops;

class GhostHandlerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // We define a standard grid cell count for our memory synchronization domain.
        total_cells = 100;
    }

    size_t total_cells;
};

/**
 * Test Case 1: Valid Buffer Synchronization & Copy Exactness
 * 
 * We populate baseline velocity and pressure fields with known numerical values:
 *     u[i] = 1.0 * i
 *     v[i] = 2.0 * i
 *     w[i] = 3.0 * i
 *     p[i] = 4.0 * i
 * 
 * When we execute sync_ghost_trial_buffers, the trial buffers (u_star, v_star, 
 * w_star, p_next) must receive exact values matching the baseline.
 */
TEST_F(GhostHandlerTest, ValidBufferSynchronization) {
    std::vector<double> u(total_cells);
    std::vector<double> v(total_cells);
    std::vector<double> w(total_cells);
    std::vector<double> p(total_cells);

    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> p_next(total_cells, 0.0);

    // We initialize baseline vectors with predictable sequential data.
    for (size_t i = 0; i < total_cells; ++i) {
        u[i] = 1.0 * i;
        v[i] = 2.0 * i;
        w[i] = 3.0 * i;
        p[i] = 4.0 * i;
    }

    // We execute the trial buffer synchronization cleaner routine.
    sync_ghost_trial_buffers(
        u.data(), v.data(), w.data(), p.data(),
        u_star.data(), v_star.data(), w_star.data(), p_next.data(),
        total_cells
    );

    // We assert that every element in the trial buffers matches the baseline data.
    for (size_t i = 0; i < total_cells; ++i) {
        EXPECT_NEAR(u_star[i], u[i], 1e-12);
        EXPECT_NEAR(v_star[i], v[i], 1e-12);
        EXPECT_NEAR(w_star[i], w[i], 1e-12);
        EXPECT_NEAR(p_next[i], p[i], 1e-12);
    }
}

/**
 * Test Case 2: Null Pointer Foundation Integrity Guard
 * 
 * If any foundation pointer (baseline or trial buffer) is passed as a null 
 * pointer, the integrity guard must intercept it and throw a std::runtime_error.
 */
TEST_F(GhostHandlerTest, NullPointerThrowsRuntimeError) {
    std::vector<double> valid_buf(total_cells, 1.0);

    // Supplying a null pointer for baseline u should trigger a contract violation.
    EXPECT_THROW({
        sync_ghost_trial_buffers(
            nullptr, valid_buf.data(), valid_buf.data(), valid_buf.data(),
            valid_buf.data(), valid_buf.data(), valid_buf.data(), valid_buf.data(),
            total_cells
        );
    }, std::runtime_error);

    // Supplying a null pointer for trial buffer u_star should trigger a contract violation.
    EXPECT_THROW({
        sync_ghost_trial_buffers(
            valid_buf.data(), valid_buf.data(), valid_buf.data(), valid_buf.data(),
            nullptr, valid_buf.data(), valid_buf.data(), valid_buf.data(),
            total_cells
        );
    }, std::runtime_error);
}
