/**
 * @file base_operator.hpp
 * @brief Abstract base interface contract for numerical solver operators.
 */

#pragma once

#include <cstddef>

namespace navier_stokes_solver {

class BaseOperator {
public:
    virtual ~BaseOperator() = default;

    /**
     * @brief Pure virtual function acting as the structural interface contract
     *        for domain-wide operator execution.
     */
    virtual void execute() = 0;
};

} // namespace navier_stokes_solver
