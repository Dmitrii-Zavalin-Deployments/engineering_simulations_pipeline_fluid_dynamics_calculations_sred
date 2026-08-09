#!/bin/bash
set -e

echo "========================================================================="
echo "=== FORENSIC AUDIT: Hydrostatic Advection Explosion Root Cause ==="
echo "========================================================================="

echo "[1] Searching for pressure gradient and advection handling in source files..."
grep -rn "grad" cpp/src/ || true
grep -rn "Advection term exploded" cpp/src/ || true

echo ""
echo "[2] Smoking-Gun Source Audit: cpp/src/predictor.cpp"
if [ -f cpp/src/predictor.cpp ]; then
    cat -n cpp/src/predictor.cpp | head -n 150
else
    echo "WARNING: cpp/src/predictor.cpp not found. Locating files..."
    find cpp/ -name "*.cpp" -o -name "*.hpp"
fi

echo ""
echo "[3] Smoking-Gun Source Audit: cpp/cpp_integration_tests/test_hydrostatic_decoupling.cpp"
if [ -f cpp/cpp_integration_tests/test_hydrostatic_decoupling.cpp ]; then
    cat -n cpp/cpp_integration_tests/test_hydrostatic_decoupling.cpp | head -n 140
else
    echo "WARNING: Test file path incorrect."
fi

echo ""
echo "========================================================================="
echo "=== AUTOMATED REPAIR TEMPLATES (COMMENCED AS COMMENTED INJECTIONS) ==="
echo "========================================================================="
# Example patch injection for correcting pressure field bindings or solver flags:
# sed -i 's/p_dynamic\[idx\] =/p[idx] =/g' cpp/cpp_integration_tests/test_hydrostatic_decoupling.cpp
# sed -i 's/double fy_val = /double fy_val = 0.0; \/\/ /g' cpp/src/predictor.cpp

echo "=== FORENSIC AUDIT COMPLETE ==="