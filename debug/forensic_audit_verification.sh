#!/usr/bin/env bash
# ============================================================================
# File: src/debug/forensic_audit.sh
# Description: Post-test forensic audit script for isolating execution failures 
#              in test_invalid_state_error_handling within the Python-C++ bridge tests.
# ============================================================================
set -euo pipefail

echo "=== [FORENSIC AUDIT] Starting diagnostic scan for test_invalid_state_error_handling ==="

# 1. Run isolated pytest diagnostic on the failing test
echo "--- Running isolated pytest diagnostic on test_invalid_state_error_handling ---"
export LD_PRELOAD=$(g++ -print-file-name=libasan.so)
export PYTHONPATH=$PWD:$PWD/build:$PYTHONPATH
pytest cpp/python_bridge_tests/test_bindings.py -k "test_invalid_state_error_handling" --tb=short || true

# 2. Cat -n for smoking-gun source audits of the test function
echo "--- Locating and auditing test_invalid_state_error_handling in test_bindings.py (cat -n) ---"
if [ -f "cpp/python_bridge_tests/test_bindings.py" ]; then
    LINE_NUM=$(grep -n "def test_invalid_state_error_handling" cpp/python_bridge_tests/test_bindings.py | cut -d: -f1 || echo "50")
    START_LINE=$((LINE_NUM - 2))
    END_LINE=$((LINE_NUM + 30))
    echo "Inspecting cpp/python_bridge_tests/test_bindings.py from line $START_LINE to $END_LINE:"
    sed -n "${START_LINE},${END_LINE}p" cpp/python_bridge_tests/test_bindings.py | cat -n
fi

echo "=== [FORENSIC AUDIT] Diagnostic audit completed successfully. ==="