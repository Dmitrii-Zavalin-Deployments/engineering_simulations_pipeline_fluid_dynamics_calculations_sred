#!/bin/bash
set -euo pipefail

echo "📌 1. Inspecting member declarations in cpp/include/orchestrator.hpp..."
python3 -c '
with open("cpp/include/orchestrator.hpp", "r") as f:
    content = f.read()
print("--- Header Member Variables ---")
for line in content.splitlines():
    if ";" in line and ("_" in line or "Config" in line or "Dimensions" in line):
        print("  ", line.strip())
'

echo "📌 2. Inspecting constructor implementation around line 56 in cpp/src/orchestrator.cpp..."
python3 -c '
with open("cpp/src/orchestrator.cpp", "r") as f:
    lines = f.readlines()
    print("--- orchestrator.cpp (Lines 40 to 70) ---")
    for i in range(39, min(70, len(lines))):
        print(f"{i+1}: {lines[i].strip()}")
'

echo "📌 3. Configuring CMake with AddressSanitizer..."
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"

echo "📌 4. Rebuilding test_mass_continuity target..."
cmake --build build --target test_mass_continuity -j$(nproc)

echo "📌 5. Running test under AddressSanitizer..."
BINARY_PATH=$(find build -name "test_mass_continuity" -type f 2>/dev/null | head -n 1)

if [ -n "$BINARY_PATH" ]; then
    export ASAN_OPTIONS="symbolize=1:detect_stack_use_after_return=1"
    "$BINARY_PATH"
else
    echo "❌ Error: test_mass_continuity binary not found."
    exit 1
fi