#!/usr/bin/env bash
# ============================================================================
# @file forensic_audit.sh
# @brief Forensic diagnostic and repair script for missing pybind11 dependencies
# ============================================================================

set -euo pipefail

echo "============================================================================"
echo "🔍 STARTING FORENSIC AUDIT: Missing pybind11 Headers"
echo "============================================================================"

# 1. Environment & Package Diagnostics
echo "--- [1] Checking Python and pybind11 Package Availability ---"
if command -v python3 &> /dev/null; then
    python3 -c "import sys; print('Python version:', sys.version)"
    python3 -c "import pybind11; print('pybind11 found via Python. Include path:', pybind11.get_include())" || echo "⚠️ pybind11 python module not installed or not in python path."
else
    echo "⚠️ python3 not found in PATH."
fi

if command -v dpkg &> /dev/null; then
    echo "--- Checking APT packages ---"
    dpkg -l | grep -E "pybind11|python3-dev" || echo "No matching APT pybind11 packages found."
fi

# 2. Build Configuration Diagnostics (Grep & Cat)
echo "--- [2] Inspecting Build Systems (CMakeLists.txt / Makefiles) ---"
find . -maxdepth 3 -name "CMakeLists.txt" -o -name "Makefile" -o -name "*.cmake" | while read -r buildfile; do
    echo "=== Scanning config file: $buildfile ==="
    grep -n -i "pybind" "$buildfile" || echo "No explicit 'pybind' reference found in $buildfile"
    echo "--- First 40 lines of $buildfile ---"
    cat -n "$buildfile" | head -n 40
done

# 3. Smoking-Gun Source Audit (cat -n)
echo "--- [3] Inspecting Offending Source Files ---"
TARGET_FILES=(
    "cpp/src/bindings.cpp"
    "cpp/src/corrector_kernel.cpp"
    "cpp/src/ppe_solver_cpp"
)

for file in "${TARGET_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "=== Smoking-Gun Source Audit: $file ==="
        grep -n "pybind11" "$file" || echo "No pybind11 include in $file"
        echo "--- Content Preview (First 25 lines) ---"
        cat -n "$file" | head -n 25
    else
        echo "ℹ️ File not found: $file"
    fi
done

echo "============================================================================"
echo "🛠️ AUTOMATED REPAIR INJECTIONS (Reference / Manual Application Guide)"
echo "============================================================================"
echo "Uncomment the appropriate sed commands below to apply automated fixes:"
echo ""

# sed -i 's|#include <pybind11/pybind11.h>|#include <pybind11/pybind11.h> // Fixed include|g' cpp/src/corrector_kernel.cpp
# sed -i 's|#include <pybind11/pybind11.h>|#include <pybind11/pybind11.h> // Fixed include|g' cpp/src/ppe_solver_cpp
# sed -i '/project(/a find_package(pybind11 REQUIRED)' CMakeLists.txt
# sed -i '/add_library(/a target_link_libraries(core_kernel PRIVATE pybind11::module)' CMakeLists.txt
# sed -i '/add_executable(unit_tests/s|cpp/src/bindings.cpp||g' CMakeLists.txt

echo "============================================================================"
echo "🏁 FORENSIC AUDIT COMPLETE"
echo "============================================================================"