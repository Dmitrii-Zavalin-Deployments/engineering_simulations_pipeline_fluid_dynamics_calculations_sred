#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Diagnostic: Locating assertion in integration test ==="
TEST_FILE="cpp/cpp_integration_tests/test_full_pipeline_accelerated_flow.cpp"
if [ -f "$TEST_FILE" ]; then
    echo "Found $TEST_FILE. Inspecting lines 260-295:"
    cat -n "$TEST_FILE" | sed -n '260,295p'
else
    echo "Searching for test_full_pipeline_accelerated_flow.cpp across workspace:"
    find . -name "test_full_pipeline_accelerated_flow.cpp" -exec cat -n {} + | sed -n '260,295p' || true
fi

echo "=== 2. Grep Audit: Tracing snap.u and prestep references ==="
git grep -n "snap.u" || true
git grep -n "FullPipelineAcceleratedFlowTest" || true
git grep -n "simulation_prestep" || true

echo "=== 3. Smoking-Gun Source Audit: Pipeline, Prestep, and Solver Logic ==="
CANDIDATE_FILES=$(git ls-files | grep -E "cpp/src|cpp/include" || true)
for file in $CANDIDATE_FILES; do
    if grep -qE "u|prestep|velocity" "$file" 2>/dev/null; then
        echo "Inspecting file: $file"
        cat -n "$file" | head -n 80
    fi
done

echo "=== 4. Automated Repair Candidates (Commented Sed) ==="
# sed -i 's/snap\.u\[idx\] = 0;/snap.u[idx] = 0.5;/g' cpp/src/simulation_prestep.cpp
# sed -i 's/initializeVelocityField(0\.0)/initializeVelocityField(0.5)/g' cpp/src/orchestrator.cpp
# sed -i 's/expected_val = 0\.0;/expected_val = 0.5;/g' cpp/cpp_integration_tests/test_full_pipeline_accelerated_flow.cpp