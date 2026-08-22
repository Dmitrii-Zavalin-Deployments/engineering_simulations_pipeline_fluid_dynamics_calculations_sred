#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo " [FORENSIC AUDIT] Navier-Stokes Core ABI & Segfault Audit"
echo "=================================================="

INTEGRATION_CMAKE="cpp/cpp_integration_tests/CMakeLists.txt"
MASS_TEST="cpp/cpp_integration_tests/test_mass_continuity.cpp"
ORCHESTRATOR_HPP="cpp/include/orchestrator.hpp"
ORCHESTRATOR_CPP="cpp/src/orchestrator.cpp"

# 1. Diagnostics: Search for library definition and preprocessor macro consistency
echo "--> Step 1: Searching for navier_stokes_core library definitions..."
grep -rn "add_library" cpp/ || true
grep -rn "navier_stokes_core" cpp/ || true

echo "--> Audit preprocessor macro presence across header files:"
grep -rn "NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS" cpp/include/ || true

# 2. Source Audit: Inspect conditional member layout in orchestrator.hpp and test implementation
echo "--> Step 2: Executing cat -n source audits..."

if [ -f "$ORCHESTRATOR_HPP" ]; then
    echo "=== Audit: $ORCHESTRATOR_HPP (Class definition & memory layout) ==="
    cat -n "$ORCHESTRATOR_HPP" | grep -n -C 5 "NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS" || true
fi

if [ -f "$MASS_TEST" ]; then
    echo "=== Audit: $MASS_TEST (Test suite setup and execution) ==="
    cat -n "$MASS_TEST" | head -n 90
fi

# 3. Automated Repair Injections
echo "--> Step 3: Providing safe automated repair injection templates..."

# # Option A: Force PUBLIC propagation of the debug macro on navier_stokes_core target so caller and library share identical binary layout
# sed -i '/add_library(navier_stokes_core/a target_compile_definitions(navier_stokes_core PUBLIC NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS)' cpp/CMakeLists.txt

# # Option B: Apply global compile definition across all CMake compilation units in integration tests
# sed -i '/cmake_minimum_required/a add_compile_definitions(NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS)' cpp/cpp_integration_tests/CMakeLists.txt

echo "=================================================="
echo " [FORENSIC AUDIT] Audit complete."
echo "=================================================="