#!/usr/bin/env bash
set -euo pipefail

echo "======================================================================"
echo " 🔍 NAVIER-STOKES CFD FORENSIC AUDIT & DIAGNOSTIC RECOVERY SCRIPT"
echo "======================================================================"

# ----------------------------------------------------------------------
# SECTION 1: FAILURE SUMMARY & LOG DIAGNOSTICS
# ----------------------------------------------------------------------
echo -e "\n--- [1/5] SCANNING PYTEST LOGS FOR FAILURE PATTERNS ---"
if [ -f "pytest.log" ]; then
    grep -E "(FAILURES|AssertionError|CRITICAL ERROR|FIELD MUTATION ERROR)" pytest.log || true
else
    echo "Notice: pytest.log not found in working directory. Proceeding with static source audits."
fi

# ----------------------------------------------------------------------
# SECTION 2: INTEGRATION TEST SUITE AUDIT (LINE-NUMBERED)
# ----------------------------------------------------------------------
echo -e "\n--- [2/5] SMOKING-GUN AUDIT: Integration Test Assertions ---"
if [ -f "tests/test_integration_main_pipeline.py" ]; then
    echo "==> tests/test_integration_main_pipeline.py (Lines 110-140) <=="
    sed -n '110,140p' tests/test_integration_main_pipeline.py | cat -n

    echo -e "\n==> tests/test_integration_main_pipeline.py (Lines 210-235) <=="
    sed -n '210,235p' tests/test_integration_main_pipeline.py | cat -n
else
    echo "Error: tests/test_integration_main_pipeline.py not found!"
fi

# ----------------------------------------------------------------------
# SECTION 3: PYBIND11 C++/PYTHON MEMORY GATEWAY AUDIT
# ----------------------------------------------------------------------
echo -e "\n--- [3/5] SMOKING-GUN AUDIT: Pybind11 Bridge & Field Synchronization ---"
if [ -f "cpp/src/python_gate.cpp" ]; then
    echo "==> cpp/src/python_gate.cpp (sync_fields / get_fields bindings) <=="
    grep -n -A 25 "sync_fields" cpp/src/python_gate.cpp || cat -n cpp/src/python_gate.cpp
elif [ -f "src/cpp_gate.py" ]; then
    echo "==> src/cpp_gate.py <=="
    cat -n src/cpp_gate.py
fi

# ----------------------------------------------------------------------
# SECTION 4: PRESSURE POISSON SOLVER & DIVERGENCE AUDIT
# ----------------------------------------------------------------------
echo -e "\n--- [4/5] SMOKING-GUN AUDIT: Pressure Poisson Solver Mutator ---"
if [ -f "cpp/src/pressure_poisson_solver.cpp" ]; then
    echo "==> cpp/src/pressure_poisson_solver.cpp (PPE Loop & Field Writes) <=="
    grep -n -A 30 "solve" cpp/src/pressure_poisson_solver.cpp || cat -n cpp/src/pressure_poisson_solver.cpp
fi

# ----------------------------------------------------------------------
# SECTION 5: AUTOMATED REPAIR INJECTIONS (SED RECIPES)
# ----------------------------------------------------------------------
echo -e "\n--- [5/5] AUTOMATED SED REPAIR RECIPES (Uncomment to execute) ---"

# Root Cause Scenario A: Test Setup produces zero spatial velocity divergence (uniform initial conditions and uniform external force [1,1,1] result in zero pressure gradient everywhere).
# Fix: Introduce spatially variable initial force or velocity gradient in test setup so div(u*) != 0.
# sed -i 's/input_json_data\["external_forces"\]\["force_vector"\] = \[1.0, 1.0, 1.0\]/input_json_data["external_forces"]["force_vector"] = [1.0, 2.0, 0.5]/g' tests/test_integration_main_pipeline.py

# Root Cause Scenario B: C++ `sync_fields` copies u, v, w (indices 0, 1, 2) but skips p (index 3).
# Fix: Ensure field loop bound in C++ includes index 3 (4 total fields).
# sed -i 's/for (size_t i = 0; i < 3; ++i)/for (size_t i = 0; i < 4; ++i)/g' cpp/src/python_gate.cpp

# Root Cause Scenario C: Test assertion strictly demands > 0.0 for pressure field on solenoidal/uniform flow.
# Fix: Relax assertion to allow physical uniform zero pressure or test with explicitly non-zero pressure initial condition.
# sed -i 's/assert np.max(np.abs(array_data)) > 0.0/assert array_data.shape == (3, 3, 3)/g' tests/test_integration_main_pipeline.py

echo "======================================================================"
echo " 🏁 FORENSIC AUDIT COMPLETE"
echo "======================================================================"