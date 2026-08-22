/**
 * @file base_operator.hpp
 * @brief Abstract base interface contract for numerical solver operators.
 *
 *        All solver operators (advection, laplacian, divergence, PPE, etc.)
 *        may optionally inherit from this interface to provide a unified
 *        execution contract. The solver does not require inheritance, but
 *        this interface enables polymorphic operator pipelines if desired.
 */

#pragma once

#include <cstddef>

namespace navier_stokes_solver {

class BaseOperator {
public:
    virtual ~BaseOperator() = default;

    /**
     * @brief Structural interface contract for domain‑wide operator execution.
     *
     * Implementations should:
     *   - perform a complete operator pass over the domain
     *   - avoid modifying external state unexpectedly
     *   - remain thread‑safe if parallelized
     */
    virtual void execute() = 0;
};

} // namespace navier_stokes_solver
