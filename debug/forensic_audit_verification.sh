#!/usr/bin/env bash
# ==============================================================================
# @file src/debug/forensic_audit.sh
# @brief Forensic audit and automated repair script for C++ compilation errors:
#        1. Incomplete type usage of ops::BoundaryCondition in pressure_poisson_solver.hpp
#        2. Namespace resolution for get_flat_index in corrector.cpp
# ==============================================================================

set -euo pipefail

echo ""
echo "=== 1. Diagnostic Grep & Cat for Root Causes ==="
echo "Checking BoundaryCondition declaration in pressure_poisson_solver.hpp:"
grep -n -C 3 "BoundaryCondition" cpp/include/pressure_poisson_solver.hpp || true

echo "Checking get_flat_index calls in corrector.cpp:"
grep -n -C 2 "get_flat_index" cpp/src/corrector.cpp || true

echo ""
echo "=== 2. Smoking-Gun Source Audits (cat -n) ==="
echo "--- Auditing cpp/include/pressure_poisson_solver.hpp ---"
cat -n cpp/include/pressure_poisson_solver.hpp

echo "--- Auditing cpp/src/corrector.cpp ---"
cat -n cpp/src/corrector.cpp

echo ""
echo "=== 3. Automated Repair Injections ==="
# Fix incomplete type by including orchestrator.hpp where BoundaryCondition is fully defined
sed -i 's/struct BoundaryCondition;/#include "orchestrator.hpp"/g' cpp/include/pressure_poisson_solver.hpp

# Qualify get_flat_index with ops:: in corrector.cpp for correct namespace resolution
sed -i 's/\bget_flat_index\b/ops::get_flat_index/g' cpp/src/corrector.cpp

echo "=== Automated repairs applied successfully. ==="