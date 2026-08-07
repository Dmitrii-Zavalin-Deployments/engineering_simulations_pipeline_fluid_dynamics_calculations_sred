// cpp/include/cell_factory.hpp

#ifndef CELL_FACTORY_HPP
#define CELL_FACTORY_HPP
namespace ops {

#include <stdexcept>
#include <string>
#include "grid_math.hpp"

namespace navier_stokes {

/**
 * @brief C++ High-Performance Cell Factory & Boundary Validator.
 * Replaces Python's src/step2/factory.py with zero-overhead inline checks 
 * and direct buffer index mapping.
 */
class CellFactory {
public:
    /**
     * @brief Validates 2-tier topology zones (Core vs Ghost) and computes 
     * the buffered flat memory index with +1 ghost padding offset.
     */
    inline static int get_buffered_index(int i, int j, int k, int nx, int ny, int nz) {
        // 1. Define valid operational zones
        bool is_core = (0 <= i && i < nx) && (0 <= j && j < ny) && (0 <= k && k < nz);
        bool is_ghost = ((-1 <= i && i <= nx) && (-1 <= j && j <= ny) && (-1 <= k && k <= nz)) && !is_core;

        // 2. Strict defensive boundary validation (matching Python factory.py hard failure)
        if (!is_core && !is_ghost) {
            throw std::out_of_range(
                "[C++ FACTORY] Out-of-bounds access: Requested cell (" + 
                std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + 
                ") is in the illegal Padding Zone (Outside [-1, " + std::to_string(nx) + "])."
            );
        }

        // 3. Map core/ghost coordinates to buffered array layout (shifting by +1 for padding layer)
        return get_flat_index(i + 1, j + 1, k + 1, nx + 2, ny + 2);
    }
};

} // namespace navier_stokes

} // namespace ops
#endif // CELL_FACTORY_HPP
