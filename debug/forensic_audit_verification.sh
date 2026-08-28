#!/usr/bin/env bash
set -euo pipefail

echo "=== [FORENSIC AUDIT] Lambda Snapshot Return Type & PipelineSnapshot Diagnostic ==="
echo "Working directory: $(pwd)"

echo "=== Searching for PipelineSnapshot definition in codebase ==="
grep -rn "PipelineSnapshot" cpp/ || echo "PipelineSnapshot symbol not found"

echo "=== Auditing test_full_pipeline_constant_flow.cpp around line 185 ==="
if [ -f "cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp" ]; then
    cat -n cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp | sed -n '175,195p'
else
    echo "test_full_pipeline_constant_flow.cpp not found!"
fi

echo "=== Checking orchestrator.hpp for snapshot container types ==="
if [ -f "cpp/include/orchestrator.hpp" ]; then
    grep -rn "snapshots" cpp/include/orchestrator.hpp -A 5 -B 5 || true
fi

echo "=== Git status ==="
git status

# sed -i 's/-> const PipelineSnapshot&//g' cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp
# sed -i 's/auto get_snapshot = \[&\](const std::string& stage_name) -> const PipelineSnapshot&/auto get_snapshot = \[&\](const std::string\& stage_name)/g' cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp