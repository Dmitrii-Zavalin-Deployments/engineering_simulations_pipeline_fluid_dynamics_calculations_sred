#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "🔍 STARTING FORENSIC AUDIT: Root Cause Analysis for Zero-Field Export Failure"
echo "=========================================================="

echo -e "\n--- [1] Codebase Diagnostic Grep: Locating Field References ---"
echo "Searching for 'fields' and velocity/pressure attributes across src/..."
grep -rn "fields" src/ || echo "Keyword 'fields' not found."
grep -rn "u" src/ --include="*.py" || echo "Attribute 'u' not found."

echo -e "\n--- [2] Smoking-Gun Source Audits (cat -n) ---"
echo "=== Auditing src/archivist.py ==="
cat -n src/archivist.py

echo -e "\n=== Auditing src/state.py ==="
cat -n src/state.py

echo -e "\n--- [3] Inspecting Recent Temporary Test Workspaces ---"
find /tmp -name "*.npy" -o -name "*.json" 2>/dev/null | tail -n 20 || echo "No temporary artifacts found."

echo -e "\n=========================================================="
echo "🛠️ AUTOMATED REPAIR PATTERNS (Inactive / Reference Only)"
echo "=========================================================="

# sed -i 's/fields = getattr(state, "fields", None)/fields = getattr(state, "fields", [getattr(state, "u", None), getattr(state, "v", None), getattr(state, "w", None), getattr(state, "p", None)])/g' src/archivist.py
# sed -i 's/if fields is not None and idx < len(fields):/if fields is not None and isinstance(fields, (list, tuple)) and idx < len(fields) and fields[idx] is not None:/g' src/archivist.py

echo "=========================================================="
echo "🚀 Forensic audit execution completed successfully."
echo "=========================================================="