// cpp/include/stencil_block.hpp

#ifndef STENCIL_BLOCK_HPP
#define STENCIL_BLOCK_HPP
namespace ops {

#include <stdexcept>
#include <string>

/**
 * @brief Logical Wiring: Represents the 7-point stencil topology in the C++ core.
 * Acts as the graph node connecting neighboring cells directly via raw pointers 
 * into the monolithic 64-byte aligned memory buffer to eliminate Python pointer-chasing overhead.
 */
struct StencilBlock {
    // Node pointers array corresponding to the 7 stencil points:
    // [0] = Center, [1] = i_minus, [2] = i_plus,
    // [3] = j_minus, [4] = j_plus, [5] = k_minus, [6] = k_plus
    double* nodes[7];

    // Physics attributes cached at assembly for performance
    double dx;
    double dy;
    double dz;
    double dt;
    double rho;
    double mu;

    /**
     * @brief Rule 4 Sync Gate: Updates the block's time-step with validation.
     */
    inline void set_dt(double value) {
        if (value <= 0.0) {
            throw std::invalid_argument("Numerical Instability: dt must be positive, got " + std::to_string(value));
        }
        dt = value;
    }

    /**
     * @brief Inline getter for time-step.
     */
    inline double get_dt() const {
        return dt;
    }

    // --- Topological Accessors (Inline for zero overhead) ---
    inline double* center() const { return nodes[0]; }
    inline double* i_minus() const { return nodes[1]; }
    inline double* i_plus() const { return nodes[2]; }
    inline double* j_minus() const { return nodes[3]; }
    inline double* j_plus() const { return nodes[4]; }
    inline double* k_minus() const { return nodes[5]; }
    inline double* k_plus() const { return nodes[6]; }

    // --- Physics Property Accessors ---
    inline double get_dx() const { return dx; }
    inline double get_dy() const { return dy; }
    inline double get_dz() const { return dz; }
    inline double get_rho() const { return rho; }
    inline double get_mu() const { return mu; }
};

} // namespace ops
#endif // STENCIL_BLOCK_HPP
