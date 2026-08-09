#pragma once

#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace navier_stokes_solver {

class BaseOperator {
public:
    virtual ~BaseOperator() = default;

    // Pure virtual function acting as the structural interface contract
    virtual void execute() = 0;
};

} // namespace navier_stokes_solver