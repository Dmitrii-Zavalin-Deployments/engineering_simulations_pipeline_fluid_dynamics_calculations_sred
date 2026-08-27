#!/usr/bin/env bash
# ============================================================================
# File: src/debug/forensic_audit.sh
# Description: Post-test forensic audit script for Navier-Stokes Python-C++ bridge 
#              diagnosing module import failures and library path resolutions.
# ============================================================================
set -euo pipefail

echo "=== [FORENSIC AUDIT] Starting diagnostic scan... ==="

# 1. Diagnostic grep/cat for code/output root causes
echo "--- Scanning test environment, Python path, and shared library locations ---"
python3 -c "import sys; print('Python executable:', sys.executable); print('Python path:', sys.path)"
find . -name "*.so" -o -name "*.pyd" -o -name "CMakeCache.txt" || echo "Warning: No C++ extension build artifacts or CMake cache found."
pip list || true
pytest --collect-only || true

# 2. Cat -n for smoking-gun source audits
echo "--- Auditing test and gateway source files (cat -n) ---"
if [ -f "cpp/python_bridge_tests/test_bindings.py" ]; then
    echo "Inspecting cpp/python_bridge_tests/test_bindings.py (lines 1 to 40):"
    sed -n '1,40p' cpp/python_bridge_tests/test_bindings.py | cat -n
fi

if [ -f "src/cpp_gate.py" ]; then
    echo "Inspecting src/cpp_gate.py (lines 1 to 30):"
    sed -n '1,30p' src/cpp_gate.py | cat -n
fi

# 3. Sed injections for automated repairs (commented out for safe dry-runs)
# # sed -i 's/import navier_stokes_cpp/import sys; sys.path.insert(0, "src"); import navier_stokes_cpp/g' cpp/python_bridge_tests/test_bindings.py
# # sed -i 's#LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"#LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"#g' CMakeLists.txt

echo "=== [FORENSIC AUDIT] Audit execution completed successfully. ==="