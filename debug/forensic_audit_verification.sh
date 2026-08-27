#!/usr/bin/env bash
# ============================================================================
# File: src/debug/forensic_audit.sh
# Description: Post-test forensic audit script for isolating execution failures 
#              in test_invalid_state_error_handling within the Python-C++ bridge tests.
# ============================================================================
set -euo pipefail

echo "=== [FORENSIC AUDIT] Starting diagnostic scan for test_invalid_state_error_handling ==="

# 1. Diagnostic grep/cat for code/output root causes (run the specific failing test with short traceback)
echo "--- Running isolated pytest diagnostic on test_invalid_state_error_handling ---"
export LD_PRELOAD=$(g++ -print-file-name=libasan.so)
export PYTHONPATH=$PWD:$PWD/build:$PYTHONPATH
pytest cpp/python_bridge_tests/test_bindings.py -k "test_invalid_state_error_handling" --tb=short || true

# 2. Cat -n for smoking-gun source audits of the test function
echo "--- Locating and auditing test_invalid_state_error_handling in test_bindings.py (cat -n) ---`"
if [ -f "cpp/python_bridge_tests/test_bindings.py" ]; then
    # Find the starting line of the failing test and print a window around it with line numbers
    LINE_NUM=$(grep -n "def test_invalid_state_error_handling" cpp/python_bridge_tests/test_bindings.py | cut -d: -f1 || echo "50")
    START_LINE=$((LINE_NUM - 2))
    END_LINE=$((LINE_NUM + 30))
    echo "Inspecting cpp/python_bridge_tests/test_bindings.py from line $START_LINE to $END_LINE:"
    sed -n "${START_LINE},${END_LINE}p" cpp/python_bridge_tests/test_bindings.py | cat -n
fi

# 3. Sed injections for automated repairs (commented out with # sed)
# # sed -i 's/with pytest.raises(ValueError):/with pytest.raises((ValueError, RuntimeError, TypeError)):/g' cpp/python_bridge_tests/test_bindings.py
# # sed -i 's/assert "Invalid state" in str(e)/# assert/g' cpp/python_bridge_tests/test_bindings.py

echo "=== [FORENSIC AUDIT] Diagnostic audit completed successfully. ==="