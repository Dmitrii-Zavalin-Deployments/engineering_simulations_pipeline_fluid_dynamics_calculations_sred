#pragma once

#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <iostream>

class BaseOperator {
public:
    virtual ~BaseOperator() = default;

    // Pure virtual function acting as the structural interface contract
    virtual void execute() = 0;
};
