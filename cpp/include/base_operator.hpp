#pragma once

class BaseOperator {
public:
    virtual ~BaseOperator() = default;

    // Pure virtual function acting as the structural interface contract
    virtual void execute() = 0;
};
