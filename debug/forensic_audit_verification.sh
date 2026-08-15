#!/usr/bin/env bash
set -euo pipefail

echo "======================================================================"
echo "🔍 FORENSIC AUDIT: C++ Engine Memory Buffer & State Sync Diagnostics"
echo "======================================================================"

# 1. Search for memory pointer pass-through and writeback in C++ Gate
echo -e "\n[1/5] Grepping C++ Gate memory references and step dispatch..."
grep -rn -E "(fields|fields\[1\]|solver|step|ascontiguousarray|ctypes|get_v|copy)" src/cpp_gate.py || true

# 2. Search for array export/save sequence in Archivist and Main
echo -e "\n[2/5] Grepping Archivist snapshot export pipeline..."
grep -rn -E "(fields|field_v|npy|save|zip|v)" src/archivist.py src/main.py || true

# 3. Line-numbered source audits of core execution files
echo -e "\n[3/5] Line-numbered audit: src/cpp_gate.py"
if [ -f "src/cpp_gate.py" ]; then
    cat -n src/cpp_gate.py
fi

echo -e "\n[4/5] Line-numbered audit: step execution loop in src/main.py"
if [ -f "src/main.py" ]; then
    cat -n src/main.py | grep -C 15 -E "(step|solver|fields|archivist|save|run)" || true
fi

# 4. Search for pytest workspace output archives for zero-value inspection
echo -e "\n[5/5] Locating temporary pytest test outputs..."
find /tmp/pytest-of-runner/ -name "navier_stokes_output.json" -o -name "*.npy" -o -name "*.zip" 2>/dev/null || true

echo -e "\n======================================================================"
echo "🛠️ REPAIR SED INJECTIONS (UNCOMMENT BEFORE EXECUTION TO APPLY)"
echo "======================================================================"

# Fix Strategy 1: Explicitly sync C++ field buffer back to state.fields after step loop in src/cpp_gate.py
# sed -i '/self._cpp_solver.step()/a \        self.state.fields[:] = self._cpp_solver.get_fields()' src/cpp_gate.py

# Fix Strategy 2: Direct view re-assignment if C++ updates native pointers in C++ gate
# sed -i 's/self.state.fields/np.ascontiguousarray(self.state.fields, dtype=np.float64)/g' src/cpp_gate.py

# Fix Strategy 3: Ensure C++ gate step passes explicit mutable state references during main step loop
# sed -i '/solver.step()/a \        self.state.v[:] = self.state.fields[1]' src/main.py