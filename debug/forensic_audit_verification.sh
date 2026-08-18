#!/usr/bin/env bash
set -euo pipefail

echo "===================================================================="
echo "[SMOKING-GUN AUDIT] Checking Pybind11 Wrapper (cpp/src/python_gate.cpp)"
echo "===================================================================="
if [ -f "cpp/src/python_gate.cpp" ]; then
    grep -n -C 10 "solve_poisson" cpp/src/python_gate.cpp || cat -n cpp/src/python_gate.cpp
fi

echo -e "\n===================================================================="
echo "[SMOKING-GUN AUDIT] Checking Python C++ Gate Interface (src/cpp_gate.py)"
echo "===================================================================="
if [ -f "src/cpp_gate.py" ]; then
    grep -n -C 10 "solve_poisson" src/cpp_gate.py || cat -n src/cpp_gate.py
fi
