#!/usr/bin/env bash
# ==============================================================================
# Forensic Audit Script for Python-C++ Binding Failures
# ==============================================================================
set -euo pipefail

echo "===================================================================="
echo "🔍 STARTING FORENSIC AUDIT: Python-C++ Binding & Exception Tracing"
echo "===================================================================="

# 1. Capture environment and pytest execution status for test_invalid_state_error_handling
echo "--- [1/4] Re-running failing test with verbose traceback ---"
python3 -m pytest cpp/python_bridge_tests/test_bindings.py -k "test_invalid_state_error_handling" -vv || true

# 2. Grep diagnostics across test files and C++ bridge for exception mismatch clues
echo "--- [2/4] Grepping exception assertions and error handling signatures ---"
echo ">>> Checking test assertions in Python test suite:"
grep -rn "with pytest.raises" cpp/python_bridge_tests/ || echo "No matches found."

echo ">>> Checking C++ error throwing / exception setting in python_gate.cpp:"
grep -rn "PyExc_" cpp/src/python_gate.cpp || echo "No PyExc_ matches found."
grep -rn "invalid_argument" cpp/src/python_gate.cpp || echo "No invalid_argument matches found."

# 3. Cat -n smoking-gun source audit for test bindings and python gate
echo "--- [3/4] Smoking-gun source audit (Line-numbered inspection) ---"
if [ -f "cpp/python_bridge_tests/test_bindings.py" ]; then
    echo ">>> Target: cpp/python_bridge_tests/test_bindings.py (test_invalid_state area)"
    cat -n cpp/python_bridge_tests/test_bindings.py | grep -C 15 -i "invalid_state" || cat -n cpp/python_bridge_tests/test_bindings.py
fi

if [ -f "cpp/src/python_gate.cpp" ]; then
    echo ">>> Target: cpp/src/python_gate.cpp (Constructor & validation checks)"
    cat -n cpp/src/python_gate.cpp | head -n 75
fi

# 4. Automated Repair Hooks (Commented Out with # sed)
echo "--- [4/4] Automated Repair Hooks (Template) ---"
# sed -i 's/pytest.raises(ValueError)/pytest.raises(RuntimeError)/g' cpp/python_bridge_tests/test_bindings.py
# sed -i 's/PyExc_ValueError/PyExc_RuntimeError/g' cpp/src/python_gate.cpp
# sed -i 's/state object cannot be None/FATAL ERROR: state object cannot be None./g' cpp/src/python_gate.cpp

echo "===================================================================="
echo "🏁 FORENSIC AUDIT COMPLETE"
echo "===================================================================="