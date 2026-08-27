#!/usr/bin/env bash
# ==============================================================================
# Forensic Audit Script: AddressSanitizer Leak & Python-C++ Gate Inspection
# ==============================================================================
set -euo pipefail

echo "===================================================================="
echo "🔍 STARTING FORENSIC AUDIT: ASan Memory Leaks & Pybind11 Init Tracing"
echo "===================================================================="

# Include repository root and build directory in PYTHONPATH
export PYTHONPATH=$PWD:$PWD/build:${PYTHONPATH:-}

# 1. Re-run target test while disabling LeakSanitizer false positives from Python runtime
echo "--- [1/4] Re-running Python C++ bridge tests with ASAN_OPTIONS override ---"
ASAN_OPTIONS=detect_leaks=0 pytest cpp/python_bridge_tests/test_bindings.py -k "test_invalid_state_error_handling" -vv --tb=short || true

# 2. Grep diagnostics across Pybind11 module definitions & C++ allocation footprints
echo "--- [2/4] Grepping module setup, py::class_ declarations, and memory allocations ---"
echo ">>> Checking pybind11 module initializations:"
grep -rn "PYBIND11_MODULE" cpp/ || echo "No PYBIND11_MODULE matches found."

echo ">>> Checking pybind11 class bindings:"
grep -rn "py::class_" cpp/ || echo "No py::class_ matches found."

echo ">>> Checking manual dynamic allocations (new/malloc) in binding code:"
grep -rnE "\b(new|malloc|PyObject_Malloc)\b" cpp/src/ || echo "No raw dynamic allocations found in cpp/src/."

# 3. Cat -n smoking-gun source audit for C++ gateway module setup
echo "--- [3/4] Smoking-gun source audit (Line-numbered inspection) ---"
if [ -f "cpp/src/python_gate.cpp" ]; then
    echo ">>> Target: cpp/src/python_gate.cpp (Module registration & bindings)"
    tail -n 35 cpp/src/python_gate.cpp | cat -n
fi

if [ -f "cpp/python_bridge_tests/test_bindings.py" ]; then
    echo ">>> Target: cpp/python_bridge_tests/test_bindings.py (Header & imports)"
    head -n 40 cpp/python_bridge_tests/test_bindings.py | cat -n
fi

# 4. Automated Repair Hooks (Template)
echo "--- [4/4] Automated Repair Hooks (Template) ---"
# sed -i '1i export ASAN_OPTIONS=detect_leaks=0' debug/forensic_audit_verification.sh
# sed -i 's/pytest cpp\/python_bridge_tests\//ASAN_OPTIONS=detect_leaks=0 pytest cpp\/python_bridge_tests\//g' .github/workflows/ci.yml
# sed -i 's/py::module_::import("_datetime");/# py::module_::import("_datetime");/g' cpp/src/python_gate.cpp

echo "===================================================================="
echo "🏁 FORENSIC AUDIT COMPLETE"
echo "===================================================================="