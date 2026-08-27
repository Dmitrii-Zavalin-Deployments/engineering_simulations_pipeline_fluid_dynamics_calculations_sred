#!/usr/bin/env bash
# ==============================================================================
# src/debug/forensic_audit.sh
# Diagnostic and automated repair script for GCOV version mismatches / return code 5
# ==============================================================================

set -uo pipefail

echo "=== [FORENSIC AUDIT] GCC & GCOV Toolchain Version Check ==="
gcc --version
gcov --version

echo "=== [FORENSIC AUDIT] Locating Stale or Mismatched Coverage Artifacts ==="
find build -name "*.gcda" -o -name "*.gcno" -ls

echo "=== [FORENSIC AUDIT] Inspecting CMake Coverage Configuration ==="
if [ -f "CMakeLists.txt" ]; then
    cat -n CMakeLists.txt | grep -i "coverage" || echo "No explicit coverage flag found in root CMakeLists.txt"
fi

# ==============================================================================
# Automated Repair Actions
# ==============================================================================
echo "=== [FORENSIC AUDIT] Purging Stale GCDA Data Files Causing GCOV Mismatch ==="
# GCOV return code 5 indicates a binary format/version mismatch between .gcno/.gcda files
# or corrupt/stale instrumentation counters left over from multi-phase CMake re-builds.
find build -name "*.gcda" -delete
echo "Stale .gcda files cleared successfully. Re-running test harness will regenerate clean counters."

# Example pattern injection for workflow or CMake fixes using commented sed commands:
# sed -i '/cmake -B build/i find build -name "*.gcda" -delete' .github/workflows/ci.yml
# sed -i 's/-DENABLE_COVERAGE=ON/-DENABLE_COVERAGE=ON -DCMAKE_CXX_COMPILER=g++/' CMakeLists.txt