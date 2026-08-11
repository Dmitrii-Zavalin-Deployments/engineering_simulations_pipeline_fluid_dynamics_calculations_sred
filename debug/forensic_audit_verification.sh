#!/usr/bin/env bash
set -euo pipefail

echo "=== FORENSIC AUDIT: Diagnostic & Root Cause Inspection ==="

echo "[1] Searching for side-effect and state mutation patterns in integration tests..."
grep -rn "side_effect" tests/integration/ || true
grep -rn "fields\[0, 0, 0, 0\]" tests/integration/ || true

echo "[2] Smoking-gun source audit with line numbers for test_03_main_cpp_gate.py:"
cat -n tests/integration/test_03_main_cpp_gate.py

# echo "[3] Executing test suite diagnostics to capture exact failure traceback:"
# pytest tests/integration/test_03_main_cpp_gate.py || true

# --- Automated Repairs via Sed Injections ---
# sed -i 's/mock_cpp_solver_instance.step.side_effect = mock_step/mock_cpp_solver_instance.step.side_effect = side_effect_step/g' tests/integration/test_03_main_cpp_gate.py