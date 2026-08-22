#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo " [FORENSIC AUDIT] Navier-Stokes Core & Integration Audit"
echo "=================================================="

INTEGRATION_CMAKE="cpp/cpp_integration_tests/CMakeLists.txt"
MASS_TEST="cpp/cpp_integration_tests/test_mass_continuity.cpp"
ORCHESTRATOR_CPP="cpp/src/orchestrator.cpp"

# 1. Diagnostics: Locate build files and scan macro propagation across targets
echo "--> Step 1: Locating build files and scanning target definitions..."
find . -name "CMakeLists.txt" -exec echo "Found CMake configuration: {}" \;

echo "--> Target linkages for navier_stokes_core:"
grep -rn "navier_stokes_core" . || true

echo "--> Macro definitions for NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS:"
grep -rn "NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS" . || true

# 2. Source Audit: Inspect CMake configuration and test source files
echo "--> Step 2: Executing cat -n source audits..."

if [ -f "$INTEGRATION_CMAKE" ]; then
    echo "=== Audit: $INTEGRATION_CMAKE ==="
    cat -n "$INTEGRATION_CMAKE"
else
    echo "WARNING: $INTEGRATION_CMAKE not found."
fi

if [ -f "$MASS_TEST" ]; then
    echo "=== Audit: $MASS_TEST (Lines 1-80) ==="
    cat -n "$MASS_TEST" | head -n 80
fi

if [ -f "$ORCHESTRATOR_CPP" ]; then
    echo "=== Audit: $ORCHESTRATOR_CPP (Macro regions) ==="
    grep -n -C 3 "NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS" "$ORCHESTRATOR_CPP" || true
fi

# 3. Automated Repair Injections
echo "--> Step 3: Providing safe automated repair injection templates..."

# # Option A: Inject global compile definition in cpp/cpp_integration_tests/CMakeLists.txt so navier_stokes_core and all tests inherit it
# sed -i '/cmake_minimum_required/a add_compile_definitions(NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS)' cpp/cpp_integration_tests/CMakeLists.txt

# # Option B: Explicitly attach public compile definition to navier_stokes_core target if defined inside cpp_integration_tests/CMakeLists.txt
# sed -i '/add_library(navier_stokes_core/a target_compile_definitions(navier_stokes_core PUBLIC NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS)' cpp/cpp_integration_tests/CMakeLists.txt

echo "=================================================="
echo " [FORENSIC AUDIT] Audit complete."
echo "=================================================="