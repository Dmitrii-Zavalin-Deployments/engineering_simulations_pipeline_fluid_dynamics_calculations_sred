#ifndef GRID_MATH_HPP
#define GRID_MATH_HPP
namespace ops {

#include <tuple>

namespace navier_stokes {

/**
 * Computes a flat index from 3D coordinates. 
 * Assumes standard row-major order: index = i + nx * j + (nx * ny) * k
 */
inline constexpr int get_flat_index(int i, int j, int k, int nx, int ny) {
    return i + (nx * j) + (nx * ny * k);
}

/**
 * SSoT Mapping: Converts flat index back to (i, j, k).
 */
inline constexpr std::tuple<int, int, int> get_coords_from_index(int index, int nx, int ny) {
    int xy_plane = nx * ny;
    
    int k = index / xy_plane;
    int rem = index % xy_plane;
    int j = rem / nx;
    int i = rem % nx;
    
    return {i, j, k};
}

} // namespace navier_stokes

} // namespace ops
#endif // GRID_MATH_HPP
