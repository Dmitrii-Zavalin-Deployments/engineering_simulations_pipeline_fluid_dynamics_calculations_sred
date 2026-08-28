#!/usr/bin/env bash
set -euo pipefail

echo "=== [FORENSIC AUDIT] Snapshot Member Field Diagnostic ==="
echo "Working directory: $(pwd)"

echo "=== Searching for incorrect .stage usages in integration tests ==="
grep -rn "snap\.stage" cpp/cpp_integration_tests/ || echo "No snap.stage matches found."

echo "=== Auditing test_full_pipeline_constant_flow.cpp around line 180-195 ==="
if [ -f "cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp" ]; then
    cat -n cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp | sed -n '180,195p'
else
    echo "Test file not found."
fi

echo "=== Verifying OrchestratorDebugSnapshot definition in orchestrator.hpp ==="
if [ -f "cpp/include/orchestrator.hpp" ]; then
    cat -n cpp/include/orchestrator.hpp | sed -n '35,48p'
fi

echo "=== Git status check ==="
git status

# sed -i 's/snap\.stage ==/snap.stage_name ==/g' cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp