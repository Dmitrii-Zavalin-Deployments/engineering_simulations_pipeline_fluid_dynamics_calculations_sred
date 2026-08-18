#!/bin/bash
# Description: Automated forensic audit for Navier-Stokes pressure field zero-mutation bug.
# Status: Active (Triggered on CI test failure)

echo "============================================================"
echo "🔍 FORENSIC AUDIT: Diagnosing 'field_p' Zero-Mutation Failure"
echo "============================================================"

echo "--- 1. Diagnostic Grep for Pressure & Field Mapping ---"
grep -rn "field_p" src/ cpp/ || echo "Notice: 'field_p' literal string not found."
grep -rn "pressure" cpp/src/ || echo "Notice: 'pressure' matches in C++ source tree."

echo "--- 2. Smoking-Gun Source Audit via cat -n ---"

if [ -f "cpp/src/pressure_poisson_solver.cpp" ]; then
    echo "=== cpp/src/pressure_poisson_solver.cpp ==="
    cat -n cpp/src/pressure_poisson_solver.cpp | head -n 120
else
    echo "❌ Target not found: cpp/src/pressure_poisson_solver.cpp"
fi

if [ -f "cpp/src/python_gate.cpp" ]; then
    echo "=== cpp/src/python_gate.cpp ==="
    cat -n cpp/src/python_gate.cpp | head -n 120
else
    echo "❌ Target not found: cpp/src/python_gate.cpp"
fi

if [ -f "src/state.py" ]; then
    echo "=== src/state.py ==="
    cat -n src/state.py | head -n 120
else
    echo "❌ Target not found: src/state.py"
fi

echo "--- 3. Automated Repair Injections (Reference Sed Rules) ---"
# sed -i 's/p_solver.solve()/p_solver.solve(fields[3])/g' cpp/src/orchestrator.cpp
# sed -i 's/memset(p, 0, size * sizeof(double))/solve_poisson_pressure(p)/g' cpp/src/pressure_poisson_solver.cpp
# sed -i 's/py::array_t<double> fields/py::array_t<double, py::array::c_style | py::array::forcecast> fields/g' cpp/src/python_gate.cpp

echo "============================================================"
echo "⚠️ Forensic audit complete. Review logs above for root cause."
