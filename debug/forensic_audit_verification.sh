#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "🔍 STARTING FORENSIC AUDIT: ASan Exception Interception Crash"
echo "=========================================================="

echo -e "\n--- 1. Environment & Runtime Diagnostics ---"
echo "Python binary: $(which python3)"
python3 --version
echo "LD_PRELOAD value: ${LD_PRELOAD:-'(not set)'}"
echo "GCC version: $(g++ --version | head -n1)"

echo -e "\n--- 2. Grep Diagnostics for Exception Handling in C++ Source ---"
echo "Searching for throw statements or exception boundaries in cpp/src/:"
grep -rn "throw " cpp/src/ || echo "No direct throw statements found via simple grep."

echo -e "\n--- 3. Smoking-Gun Source Audit (cpp/src/python_gate.cpp around line 34) ---"
if [ -f "cpp/src/python_gate.cpp" ]; then
    echo "Displaying lines 1 to 65 of cpp/src/python_gate.cpp with line numbers:"
    cat -n cpp/src/python_gate.cpp | sed -n '1,65p'
else
    echo "⚠️ Warning: cpp/src/python_gate.cpp not found at expected path."
fi

echo -e "\n--- 4. Automated Repair Simulation & Instructions ---"
echo "To fix the ASan __cxa_throw mismatch when running against a non-ASan Python interpreter,"
echo "wrap the constructor body in a try-catch block or handle pybind11 exception translation."
echo ""
echo "Example # sed injection for automated repair (commented out):"
# sed -i '/PythonSolverBridge::PythonSolverBridge/ { N; s/{/{\n    try {/}' cpp/src/python_gate.cpp
# sed -i '/^}/i \    } catch (const std::exception& e) { PyErr_SetString(PyExc_RuntimeError, e.what()); throw_error_already_set(); }' cpp/src/python_gate.cpp

echo "=========================================================="
echo "🏁 FORENSIC AUDIT COMPLETE"
echo "=========================================================="