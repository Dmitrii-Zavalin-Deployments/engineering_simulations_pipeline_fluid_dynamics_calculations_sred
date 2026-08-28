#!/usr/bin/env bash
set -euo pipefail

echo "=== [FORENSIC AUDIT] Fixing OrchestratorDebugSnapshot member reference ==="
echo "Working directory: $(pwd)"

echo "=== Searching for incorrect .stage property usage ==="
grep -rn "snap\.stage" cpp/cpp_integration_tests/ || echo "No occurrences found"

echo "=== Auditing test_full_pipeline_constant_flow.cpp around line 187 ==="
if [ -f "cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp" ]; then
    cat -n cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp | sed -n '180,195p'
else
    echo "Test file not found!"
fi

echo "=== Git status check ==="
git status

# sed -i 's/snap\.stage ==/snap.stage_name ==/g' cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp