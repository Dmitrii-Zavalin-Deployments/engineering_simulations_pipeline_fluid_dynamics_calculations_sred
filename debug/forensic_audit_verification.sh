#!/bin/bash
set -euo pipefail

TARGET_FILE="cpp/cpp_integration_tests/test_mass_continuity.cpp"

echo "📌 1. Patching $TARGET_FILE to strip hot-loop forensic loggers..."

# Remove per-cell debug logging statements from the 3D iteration loop
sed -i '/\[debug\] Cell (i=/d' "$TARGET_FILE"
sed -i '/\[debug\] Neighbors:/d' "$TARGET_FILE"
sed -i '/\[debug\] div_u=/d' "$TARGET_FILE"
sed -i '/<< " dudx="/d' "$TARGET_FILE"
sed -i '/<< " dvdy="/d' "$TARGET_FILE"
sed -i '/<< " dwdz="/d' "$TARGET_FILE"

echo "📌 2. Rebuilding C++ integration test target..."
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target test_mass_continuity -j$(nproc)

echo "📌 3. Running Mass Continuity Integration Test under AddressSanitizer..."
./build/cpp/cpp_integration_tests/test_mass_continuity