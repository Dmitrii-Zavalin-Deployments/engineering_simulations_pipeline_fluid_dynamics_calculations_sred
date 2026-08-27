#!/bin/bash
set -eo pipefail

echo "=================================================================="
echo "    NAVIER-STOKES SOLVER: PATH-AWARE C++ STACK AUDIT            "
echo "=================================================================="

# 1. Scan all C++ source and header files under cpp/ for local stack arrays (fixed or VLAs)
echo "[*] Step 1: Scanning for stack array allocations across cpp/ directory..."
echo "------------------------------------------------------------------"
grep -rnE "\b(double|float|int|uint[0-9]+_t|size_t|char)\s+\w+\s*\[[^\]]+\]" cpp/ || echo "No explicit bracket array declarations found."

# 2. Inspect test file implementation specifically for stack buffers or raw arrays
echo ""
echo "[*] Step 2: Auditing local stack variables in integration & unit tests..."
echo "------------------------------------------------------------------"
grep -rnE "(double|float|int|char|uint[0-9]+_t)\s+\w+\[" cpp/cpp_integration_tests/ cpp/cpp_unit_tests/ || echo "No local arrays declared in test suites."

# 3. Check for unsafe memory operations or raw pointer writes
echo ""
echo "[*] Step 3: Checking for raw pointer decay or unsafe memory routines..."
echo "------------------------------------------------------------------"
grep -rnE "\b(memcpy|memset|strcpy|sprintf|snprintf)\b" cpp/ || echo "No unsafe C-style memory operations found."

# 4. Target core solver cpp files directly
echo ""
echo "[*] Step 4: Full audit of cpp/src files..."
echo "------------------------------------------------------------------"
find cpp/src -type f \( -name "*.cpp" -o -name "*.hpp" \) -exec grep -HnE "\b(double|float|int)\s+\w+\s*\[" {} + || echo "Clean: No raw array declarations found in cpp/src."

echo "=================================================================="
echo "                  AUDIT COMPLETE - REVIEW OUTPUT                  "
echo "=================================================================="