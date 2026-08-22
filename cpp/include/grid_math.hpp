/**
 * @file grid_math.hpp
 * @brief Single Source of Truth (SSoT) for grid dimensions and
 *        row‑major 3D → 1D indexing used consistently across the solver.
 *
 *        Flat index layout (C++ row‑major):
 *            index = i + nx * j + (nx * ny) * k
 *
 *        This matches the corrected solver operators and ensures
 *        consistent memory access across all modules.
 */

#ifndef GRID_MATH_HPP
#define GRID_MATH_HPP

#include <tuple>
#include <stdexcept>

namespace navier_stokes_solver {

/**
 * @brief Structure representing the grid dimensions and spacing.
 */
struct GridDimensions {
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double z_min;
    double z_max;
    int nx;
    int ny;
    int nz;
    double dx;
    double dy;
    double dz;

    // Basic safety validation
    void validate() const {
        if (nx <= 0 || ny <= 0 || nz <= 0) {
            throw std::invalid_argument("GridDimensions ERROR: nx, ny, nz must be positive.");
        }
        if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
            throw std::invalid_argument("GridDimensions ERROR: dx, dy, dz must be strictly positive.");
        }
    }
};

/**
 * @brief Computes a flat index from 3D coordinates using standard row‑major order.
 *
 * Layout:
 *   i varies fastest
 *   j is next
 *   k is slowest
 *
 * This is the ONLY valid indexing scheme for the solver.
 */
inline constexpr int get_flat_index(int i, int j, int k, int nx, int ny) {
    return i + (nx * j) + (nx * ny * k);
}

/**
 * @brief Converts a flat index back to (i, j, k) coordinates.
 */
inline constexpr std::tuple<int, int, int> get_coords_from_index(int index, int nx, int ny) {
    int xy_plane = nx * ny;
    int k = index / xy_plane;
    int rem = index % xy_plane;
    int j = rem / nx;
    int i = rem % nx;
    return {i, j, k};
}

} // namespace navier_stokes_solver

#endif // GRID_MATH_HPP

