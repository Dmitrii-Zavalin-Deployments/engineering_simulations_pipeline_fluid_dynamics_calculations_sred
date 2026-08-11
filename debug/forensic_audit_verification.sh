#!/usr/bin/env bash
set -euo pipefail

echo "=== FORENSIC AUDIT: Diagnostic & Root Cause Inspection ==="

echo "[1] Searching for tensor field mutations, initializations, and side-effect patterns..."
grep -rn "fields" src/ tests/integration/ || true
grep -rn "side_effect" tests/integration/ || true
grep -rn "99.9" src/ tests/integration/ || true

echo "[2] Smoking-gun source audit with line numbers for test_03_main_cpp_gate.py:"
cat -n tests/integration/test_03_main_cpp_gate.py

echo "[3] Smoking-gun source audit with line numbers for cpp_gate.py:"
cat -n src/cpp_gate.py || true

# echo "[4] Executing targeted pytest suite with verbose tracing to isolate float mismatch traceback:"
# pytest tests/integration/test_03_main_cpp_gate.py -vv || true

# --- Automated Repairs via Sed Injections ---
# sed -i 's/state.fields\[0, 0, 0, 0\] = 99.9/state.fields[0, 0, 0, 0] = 99.9/g' tests/integration/test_03_main_cpp_gate.py