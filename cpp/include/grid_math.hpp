/**
 * @file grid_math.hpp
 * @brief Header for grid dimensions and coordinate indexing utilities (Single Source of Truth).
 *        Updated to align C++ flat indexing with NumPy C-contiguous memory strides where 
 *        the z-axis (k) varies fastest.
 */

#ifndef GRID_MATH_HPP
#define GRID_MATH_HPP

#include <tuple>

namespace navier_stokes_solver {

/**
 * @brief Structure representing the grid dimensions and spacing.
 */
struct GridDimensions {
    int nx;
    int ny;
    int nz;
    double dx;
    double dy;
    double dz;
};

/**
 * @brief Computes a flat index from 3D coordinates matching NumPy C-contiguous layout.
 * Assumes C-contiguous order: index = i * (ny * nz) + j * nz + k (k varies fastest).
 */
inline constexpr int get_flat_index(int i, int j, int k, int ny, int nz) {
    return (i * ny * nz) + (j * nz) + k;
}

/**
 * @brief SSoT Mapping: Converts flat index back to (i, j, k) for C-contiguous layout.
 */
inline constexpr std::tuple<int, int, int> get_coords_from_index(int index, int ny, int nz) {
    int yz_plane = ny * nz;
    int i = index / yz_plane;
    int rem = index % yz_plane;
    int j = rem / nz;
    int k = rem % nz;
    return {i, j, k};
}

} // namespace navier_stokes_solver

#endif // GRID_MATH_HPP
