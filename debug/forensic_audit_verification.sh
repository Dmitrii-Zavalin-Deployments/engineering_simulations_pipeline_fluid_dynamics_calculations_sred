#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo "🚀 STARTING FORENSIC AUDIT: SOLID NEUMANN FAILURE"
echo "=================================================="

# 1. Diagnostic: Locate the test definition and assertion causing the failure
echo -n "--- [1] Locating SolidNeumannBoundaryZeroNormalGradient Test ---"
grep -rn "SolidNeumannBoundaryZeroNormalGradient" cpp/ || echo "Test reference not found."

# 2. Smoking-Gun Source Audit: Inspect test_solid_neumann_pressure.cpp around line 64
echo -n "--- [2] Inspecting test_solid_neumann_pressure.cpp ---"
if [ -f "cpp/cpp_unit_tests/test_solid_neumann_pressure.cpp" ]; then
    cat -n cpp/cpp_unit_tests/test_solid_neumann_pressure.cpp | sed -n '40,75p'
else
    echo "Warning: cpp/cpp_unit_tests/test_solid_neumann_pressure.cpp not found."
fi

# 3. Smoking-Gun Source Audit: Inspect apply_solid_neumann_pressure_parallel in pressure_poisson_solver.cpp
echo -n "--- [3] Inspecting apply_solid_neumann_pressure_parallel Implementation ---"
if [ -f "cpp/src/pressure_poisson_solver.cpp" ]; then
    grep -n -C 30 "apply_solid_neumann_pressure_parallel" cpp/src/pressure_poisson_solver.cpp
else
    echo "Warning: cpp/src/pressure_poisson_solver.cpp not found."
fi


# 5. Automated Repair Injections (Commented out for reference/safety)
# # sed -i 's/if (mask\[idx\] == 1) continue;/\\/ /g' cpp/src/pressure_poisson_solver.cpp
# # sed -i 's/for (int k = 1; k < nz - 1; ++k)/for (int k = 0; k < nz; ++k)/g' cpp/src/pressure_poisson_solver.cpp

echo "=================================================="
echo "🏁 FORENSIC AUDIT COMPLETE"
echo "=================================================="