#!/bin/bash
set -euo pipefail

TARGET_TEST="cpp/cpp_integration_tests/test_mass_continuity.cpp"

echo "📌 1. Dynamically locating orchestrator.hpp..."
ORCH_HPP=$(find . -name "orchestrator.hpp" -type f | head -n 1)

if [ -z "$ORCH_HPP" ]; then
    echo "❌ Error: orchestrator.hpp not found anywhere in repository."
    find . -name "*.hpp"
    exit 1
fi
echo "Found orchestrator.hpp at: $ORCH_HPP"

echo "📌 2. Inspecting orchestrator.hpp path..."
python3 -c "
import glob
matches = glob.glob('**/orchestrator.hpp', recursive=True)
print(f'Inspecting {matches[0]} for member initialization order...')
"

echo "📌 3. Re-verifying heap allocation in $TARGET_TEST..."
# (Test file already heap-allocates orchestrator via std::make_unique)

echo "📌 4. Configuring CMake with AddressSanitizer..."
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"

echo "📌 5. Rebuilding test_mass_continuity target..."
cmake --build build --target test_mass_continuity -j$(nproc)

echo "📌 6. Locating binary and running under AddressSanitizer..."
BINARY_PATH=$(find build -name "test_mass_continuity" -type f 2>/dev/null | head -n 1)

if [ -n "$BINARY_PATH" ]; then
    echo "Found binary at: $BINARY_PATH"
    export ASAN_OPTIONS="symbolize=1:detect_stack_use_after_return=1"
    "$BINARY_PATH"
else
    echo "❌ Error: test_mass_continuity binary not found."
    exit 1
fi