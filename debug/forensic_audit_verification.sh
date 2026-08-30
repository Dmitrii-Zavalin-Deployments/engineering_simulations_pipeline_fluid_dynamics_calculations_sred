#!/usr/bin/env bash
# ==============================================================================
# @file src/debug/forensic_audit.sh
# @brief Forensic Diagnostic & Remediation Script for BoundaryValues Layout & Narrowing Conversion
# ==============================================================================

set -euo pipefail

echo "=== [STEP 1] Diagnostic Search for BoundaryValues Definition ==="
grep -rn "struct BoundaryValues" cpp/include/ || echo "Search complete."

echo "=== [STEP 2] Smoking-Gun Source Audit: boundary_condition.hpp ==="
if [ -f "cpp/include/boundary_condition.hpp" ]; then
    cat -n cpp/include/boundary_condition.hpp
fi

echo "=== [STEP 3] Smoking-Gun Source Audit: test_simulation_prestep.cpp ==="
if [ -f "cpp/cpp_unit_tests/test_simulation_prestep.cpp" ]; then
    cat -n cpp/cpp_unit_tests/test_simulation_prestep.cpp
fi

echo "=== [STEP 4] Build Logs and Artifact Check ==="
find . -name "*.log" -o -name "CMakeCache.txt" 2>/dev/null || true

# ==============================================================================
# REMEDIATION INJECTIONS (Commented out with # sed as requested)
# ==============================================================================
# sed -i '/struct BoundaryValues {/{n;s/bool /double /g}' cpp/include/boundary_condition.hpp
# sed -i 's/inflow_bc.values = {[^}]*};/inflow_bc.values.u = 1.0; inflow_bc.values.v = 0.5; inflow_bc.values.w = 0.2; inflow_bc.values.p = 10.0;/g' cpp/cpp_unit_tests/test_simulation_prestep.cpp
# sed -i 's/wall_bc.values = {[^}]*};/wall_bc.values.u = 0.0; wall_bc.values.v = 0.0; wall_bc.values.w = 0.0; wall_bc.values.p = 0.0;/g' cpp/cpp_unit_tests/test_simulation_prestep.cpp
# sed -i 's/unknown_bc.values = {[^}]*};/unknown_bc.values.u = 0.0; unknown_bc.values.v = 0.0; unknown_bc.values.w = 0.0; unknown_bc.values.p = 101325.0;/g' cpp/cpp_unit_tests/test_simulation_prestep.cpp
# sed -i 's/free_slip_bc.values = {[^}]*};/free_slip_bc.values.u = 0.0; free_slip_bc.values.v = 1.5; free_slip_bc.values.w = 0.0; free_slip_bc.values.p = 0.0;/g' cpp/cpp_unit_tests/test_simulation_prestep.cpp
# sed -i 's/pressure_bc.values = {[^}]*};/pressure_bc.values.u = 0.0; pressure_bc.values.v = 0.0; pressure_bc.values.w = 0.0; pressure_bc.values.p = 101325.0;/g' cpp/cpp_unit_tests/test_simulation_prestep.cpp