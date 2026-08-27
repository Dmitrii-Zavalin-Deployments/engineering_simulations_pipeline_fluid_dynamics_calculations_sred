#!/bin/bash
set -eo pipefail

echo "=================================================================="
echo "    NAVIER-STOKES SOLVER: ADVANCED VLA & STACK BUFFER AUDIT     "
echo "=================================================================="

# 1. Hunt for Variable-Length Arrays (VLAs) and non-literal stack allocations
echo "[*] Step 1: Scanning for Variable-Length Arrays (VLAs) [e.g., arr[nx], arr[total_cells]]..."
echo "------------------------------------------------------------------"
grep -rnE "\b(double|float|int|char|size_t)\s+\w+\s*\[\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\]" src/ include/ tests/ || echo "No explicit VLA patterns detected with single identifiers."
grep -rnE "\b(double|float|int|char|size_t)\s+\w+\s*\[\s*[^0-9\]]+\s*\]" src/ include/ tests/ || echo "No complex non-literal bracket expressions found."

# 2. Hunt for raw pointer casting or stack array passing to C-style functions
echo ""
echo "[*] Step 2: Scanning for raw pointer decays and potential buffer overflows..."
echo "------------------------------------------------------------------"
grep -rnE "\b(memcpy|memset|strcpy|sprintf)\s*\(" src/ include/ || echo "No unsafe C-string/memory functions found."

# 3. Smoking-Gun Source Audit: Inspect solver inner modules for local arrays
echo ""
echo "[*] Step 3: Inspecting core solver implementation files..."
echo "------------------------------------------------------------------"
for file in src/pressure_poisson_solver.cpp src/orchestrator.cpp src/advection.cpp src/laplacian.cpp; do
    if [ -f "$file" ]; then
        echo "=== FILE: $file ==="
        grep -nE "(double|float|int)\s+\w+\s*\[" "$file" || echo "No stack array declarations found in $file."
    fi
done

echo "=================================================================="
echo "                  AUDIT COMPLETE - REVIEW OUTPUT                  "
echo "=================================================================="