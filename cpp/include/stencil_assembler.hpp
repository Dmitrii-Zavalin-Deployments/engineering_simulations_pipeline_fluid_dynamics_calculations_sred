// cpp/include/stencil_assembler.hpp

#ifndef STENCIL_ASSEMBLER_HPP
#define STENCIL_ASSEMBLER_HPP
namespace ops {

#include <vector>
#include "stencil_block.hpp"
#include "grid_math.hpp"

namespace navier_stokes {

/**
 * @brief High-performance C++ Stencil Assembler.
 * Replaces Python's src/step2/stencil_assembler.py and CellRegistry by directly 
 * constructing contiguous StencilBlocks pointing into the raw NumPy memory buffer.
 */
std::vector<StencilBlock> assemble_stencil_matrix_cpp(
    double* raw_fields_ptr,
    int num_fields,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho, double mu
);

} // namespace navier_stokes

} // namespace ops
#endif // STENCIL_ASSEMBLER_HPP
