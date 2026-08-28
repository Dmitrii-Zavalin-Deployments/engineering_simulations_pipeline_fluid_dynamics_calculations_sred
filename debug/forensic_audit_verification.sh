#!/usr/bin/env bash
set -euo pipefail

echo "=== [FORENSIC AUDIT] OrchestratorDebugSnapshot Structure Diagnostic ==="
echo "Working directory: $(pwd)"

echo "=== Searching for OrchestratorDebugSnapshot definition in headers ==="
find cpp -name "*.hpp" -exec grep -Hn "struct OrchestratorDebugSnapshot" {} + || \
find cpp -name "*.hpp" -exec grep -Hn "class OrchestratorDebugSnapshot" {} + || \
grep -rn "OrchestratorDebugSnapshot" cpp/include/ || true

echo "=== Auditing orchestrator.hpp around snapshot definition ==="
if [ -f "cpp/include/orchestrator.hpp" ]; then
    grep -n "struct OrchestratorDebugSnapshot" cpp/include/orchestrator.hpp -A 20 || \
    cat -n cpp/include/orchestrator.hpp | head -n 120
else
    find cpp -name "orchestrator.hpp" -exec cat -n {} +
fi

echo "=== Auditing test_full_pipeline_constant_flow.cpp get_snapshot implementation ==="
if [ -f "cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp" ]; then
    grep -n "get_snapshot" cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp -A 15 -B 5
fi

echo "=== Git status check ==="
git status

# sed -i 's/snap\.stage/snap.name/g' cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp
# sed -i 's/snap\.stage/snap.stage_name/g' cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp
# sed -i 's/if (snap\.stage == stage_name)/if (snapshots[\&stage_name - \&stage_name] ...)/g' cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp