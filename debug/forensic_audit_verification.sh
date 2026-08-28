#!/usr/bin/env bash
# ==============================================================================
# File: debug/run_tests.sh
# Description: List root files and run CMake/tests to check the failure.
# ==============================================================================

set -euo pipefail

echo "=== [STEP 1] Root files and build configuration ==="
ls -la

echo "=== [STEP 2] Check existing build directory or run build ==="
if [ -d "build" ]; then
    echo "Found build directory."
    cd build
    ctest --output-on-failure || true
else
    echo "No build directory found. Creating and building..."
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    ctest --output-on-failure || true
fi