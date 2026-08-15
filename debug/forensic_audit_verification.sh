#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================================="
echo "       FORENSIC AUDIT & MEMORY MUTATION DIAGNOSTIC SUITE"
echo "=========================================================================="

# [1] Inspect C++ Python gate binding expectations for SolverState attributes
echo -e "\n[1] INSPECTING C++ / PYTHON BRIDGE ATTRIBUTE EXPECTATIONS"
echo "--------------------------------------------------------------------------"
grep -rn -E "(external_forces|gravity|fields|sync_fields|u_val|v_val|w_val|scalar_p|step)" src/ cpp/ include/ 2>/dev/null || true

# [2] Inspect actual array values inside exported test workspace NPY files
echo -e "\n[2] INSPECTING EXPORTED NPY FIELD SNAPSHOT VALUES IN /tmp"
echo "--------------------------------------------------------------------------"
python3 -c '
import pathlib, numpy as np
for path in sorted(pathlib.Path("/tmp").rglob("field_*.npy")):
    arr = np.load(path)
    print(f"File: {path} | Shape: {arr.shape} | Min: {arr.min():.6f} | Max: {arr.max():.6f} | NonZero: {np.count_nonzero(arr)}")
' 2>/dev/null || true

# [3] Source Audit: src/cpp_gate.py
echo -e "\n[3] SMOKING-GUN SOURCE AUDIT: src/cpp_gate.py"
echo "--------------------------------------------------------------------------"
if [ -f "src/cpp_gate.py" ]; then
    cat -n src/cpp_gate.py
fi

# [4] Source Audit: src/state.py
echo -e "\n[4] SMOKING-GUN SOURCE AUDIT: src/state.py"
echo "--------------------------------------------------------------------------"
if [ -f "src/state.py" ]; then
    cat -n src/state.py
fi

# [5] Source Audit: C++ Bridge implementation (python_gate.cpp / orchestrator.cpp)
echo -e "\n[5] SMOKING-GUN SOURCE AUDIT: C++ GATE SOURCE FILES"
echo "--------------------------------------------------------------------------"
find . -name "python_gate.cpp" -o -name "orchestrator.cpp" | xargs -r cat -n 2>/dev/null || true

echo -e "\n=========================================================================="
echo "                      AUTOMATED REPAIR SED INJECTIONS"
echo "=========================================================================="

# Inject gravity vector property directly onto SolverState for C++ attribute lookup
# sed -i '/self.external_forces/a \        self.gravity = self.external_forces.get("gravity_vector", [0.0, -9.81, 0.0])' src/state.py

# Inject explicit array sync call right after step_simulation inside main loop
# sed -i '/step_simulation(state)/a \            if hasattr(state, "_cpp_solver") and hasattr(state._cpp_solver, "sync_fields"): state._cpp_solver.sync_fields(state)' src/main.py

# Ensure fields array is strictly C-contiguous float64 before passing to C++ gate
# sed -i 's/solver.step(state)/state.fields = np.ascontiguousarray(state.fields, dtype=np.float64)\n        solver.step(state)/g' src/cpp_gate.py