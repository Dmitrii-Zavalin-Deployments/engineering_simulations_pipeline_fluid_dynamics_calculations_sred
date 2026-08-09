#!/usr/bin/env bash
# ==============================================================================
# @file src/debug/forensic_audit.sh
# @brief Forensic audit and automated repair script for C++ compilation errors:
#        1. Incomplete type usage of ops::BoundaryCondition in pressure_poisson_solver.hpp
#        2. Namespace resolution for get_flat_index in corrector.cpp
# ==============================================================================

set -euo pipefail

echo "================================================================Cache & Build Environment Diagnostics ==="
uname -a
cmake --version || true
gcc --version || true

echo ""
echo "=== 1. Diagnostic Grep & Cat for Root Causes ==="
echo "Checking BoundaryCondition definition vs header inclusions:"
grep -n -C 3 "BoundaryCondition" cpp/include/pressure_poisson_solver.hpp || true

echo "Checking namespace and get_flat_index scope in corrector.cpp:"
grep -n -C 3 "get_flat_index" cpp/src/corrector.cpp || true

echo ""
echo "=== 2. Smoking-Gun Source Audits (cat -n) ==="
echo "--- Auditing cpp/include/pressure_poisson_solver.hpp ---"
cat -n cpp/include/pressure_poisson_solver.hpp

echo "--- Auditing cpp/src/corrector.cpp ---"
cat -n cpp/src/corrector.cpp

echo ""
echo "=== 3. Automated Repair Injections (Commented via # sed) ==="
# Include orchestrator.hpp to supply full definition of BoundaryCondition
# sed -i 's/struct BoundaryCondition;/#include "orchestrator.hpp"/g' cpp/include/pressure_poisson_solver.hpp

# Qualify get_flat_index with ops:: in corrector.cpp
# sed -i 's/\bget_flat_index\b/ops::get_flat_index/g' cpp/src/corrector.cpp

echo "=== Forensic audit script completed successfully. ==="