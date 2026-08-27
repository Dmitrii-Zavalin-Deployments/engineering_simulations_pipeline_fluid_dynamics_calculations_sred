#!/usr/bin/env bash
# ============================================================================
# Forensic Audit Script for Navier-Stokes Python-C++ Bridge Test Failures
# ============================================================================
set -euo pipefail

echo "=== [FORENSIC AUDIT] Starting diagnostic scan... ==="

# 1. Diagnostic grep/cat for code/output root causes
echo "--- Scanning test environment and logs ---"
python3 -c "import sys; print('Python executable:', sys.executable); print('Python path:', sys.path)"
pip list || true

echo "--- Checking for import errors or build artifacts in python bridge tests ---"
find cpp/python_bridge_tests -name "*.py" -exec python3 -m py_compile {} \; 2>&1 || true

# 2. Cat -n for smoking-gun source audits
echo "--- Auditing source files (cat -n) ---"
if [ -f "cpp/python_bridge_tests/test_python_gate.py" ]; then
    echo "Inspecting test_python_gate.py:"
    cat -n cpp/python_bridge_tests/test_python_gate.py
fi

if [ -f "cpp/python_bridge_tests/test_stride_alignment.py" ]; then
    echo "Inspecting test_stride_alignment.py:"
    cat -n cpp/python_bridge_tests/test_stride_alignment.py
fi

# 3. Sed injections for automated repairs (commented out for safety validation)
# # sed -i 's/navier_stokes_cpp is None/False/g' cpp/python_bridge_tests/test_python_gate.py
# # sed -i 's/pytest.skip(/# pytest.skip(/g' cpp/python_bridge_tests/test_python_gate.py

echo "=== [FORENSIC AUDIT] Audit execution completed successfully. ==="