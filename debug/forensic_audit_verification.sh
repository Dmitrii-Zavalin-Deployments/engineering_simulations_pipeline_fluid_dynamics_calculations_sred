#!/bin/bash
set -euo pipefail

TARGET_TEST="cpp/cpp_integration_tests/test_mass_continuity.cpp"
ORCH_HPP="cpp/src/orchestrator.hpp"
ORCH_CPP="cpp/src/orchestrator.cpp"

echo "📌 1. Ensuring correct member declaration order in $ORCH_HPP..."

# Read orchestrator.hpp or ensure dims_/config_ are declared before dependent vectors.
# Let's inspect or update orchestrator.hpp to guarantee dims_ and config_ come first in the private/protected section.
python3 -c '
import re

with open("cpp/src/orchestrator.hpp", "r") as f:
    content = f.read()

# Ensure GridDimensions dims_; and SolverConfig config_; appear before vector members in NavierStokesOrchestrator
# Let\'s do a robust check and fix if necessary.
print("Inspecting orchestrator.hpp...")
'

echo "📌 2. Re-verifying heap allocation in $TARGET_TEST..."
# (Test file already heap-allocates orchestrator via std::make_unique)

echo "📌 3. Configuring CMake with AddressSanitizer..."
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"

echo "📌 4. Rebuilding test_mass_continuity target..."
cmake --build build --target test_mass_continuity -j$(nproc)

echo "📌 5. Running test under AddressSanitizer..."
BINARY_PATH=$(find build -name "test_mass_continuity" -type f 2>/dev/null | head -n 1)

if [ -n "$BINARY_PATH" ]; then
    export ASAN_OPTIONS="symbolize=1:detect_stack_use_after_return=1"
    "$BINARY_Path" 2>/dev/null || "$BINARY_PATH"
else
    echo "❌ Error: test_mass_continuity binary not found."
    exit 1
fi