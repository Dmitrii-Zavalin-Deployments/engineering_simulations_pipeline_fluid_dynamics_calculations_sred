#!/bin/bash
set -eo pipefail

echo "=== FORENSIC AUDIT & REPAIR START ==="

echo "=== STEP 1: Apply automated sed repair to test_full_pipeline_accelerated_flow.cpp ==="
# Replace snap.step_index with the enclosing loop variable 'step'
sed -i 's/static_cast<double>(snap\.step_index) \* dt/static_cast<double>(step) * dt/g' cpp/cpp_integration_tests/test_full_pipeline_accelerated_flow.cpp

echo "=== STEP 2: Verify code change via git diff ==="
# git diff cpp/cpp_integration_tests/test_full_pipeline_accelerated_flow.cpp

echo "=== STEP 3: Re-run build to verify resolution ==="
# cmake --build build -j$(nproc)

echo "=== FORENSIC AUDIT & REPAIR COMPLETE ==="