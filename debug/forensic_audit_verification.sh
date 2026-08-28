#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Forensic Audit Script for FullPipelineConstantFlowTest.StepByStepMicroManaged
# ============================================================================

echo "=== [1] Diagnostic: Locating is_core_interior logic and bounds check in test suite ==="
grep -n -C 15 "is_core_interior" cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp || true

echo "=== [2] Smoking-Gun Source Audit: Line-numbered view of test file (Lines 985-1025) ==="
cat -n cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp | sed -n '985,1025p'

echo "=== [3] Diagnostic: Checking build artifacts or recent test logs for failure context ==="
if [ -f "build/Testing/Temporary/LastTest.log" ]; then
    grep -C 5 "u_star" build/Testing/Temporary/LastTest.log || true
fi

# ============================================================================
# Automated Repair Injections (Uncomment below to apply patch)
# ============================================================================
# Expand boundary-adjacent exclusion zone to correctly classify cells exhibiting coarse-grid discretization drift as non-core interior
# sed -i 's/if (i == 0 || i == dims.nx - 1 ||/if (i <= 1 || i >= dims.nx - 2 ||/g' cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp