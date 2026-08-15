#!/usr/bin/env bash
# ==============================================================================
# src/debug/forensic_audit.sh
# Automated Forensic Audit & Diagnostic Script for CI/CD Pipelines
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "[FORENSIC AUDIT] Starting Pipeline & Environment Diagnostic Suite..."
echo "=============================================================================="

echo "--- 1. Python Environment & Installed Packages ---"
python3 -m pip list || true

echo "--- 2. Checking Dependency & Configuration Files ---"
find . -maxdepth 2 -name "*requirements*" -o -name "pyproject.toml" -o -name "setup.py" || true
cat pyproject.toml 2>/dev/null || cat requirements.txt 2>/dev/null || true

echo "--- 3. Smoking-Gun Source Audit via cat -n (src/main.py) ---"
cat -n src/main.py || true

echo "--- 4. Checking Test Invocations of main() ---"
grep -rn "def test_main" tests/ || true

echo "--- 5. Automated Repair Instructions (Sed Injections) ---"
# To restore clean architectural separation (keeping sys.exit strictly in if __name__ == '__main__'):
# sed -i '/sys.exit(0)/d' src/main.py
# sed -i '/sys.exit(1)/d' src/main.py

echo "=============================================================================="
echo "[FORENSIC AUDIT] Diagnostic Audit Completed Successfully."
echo "=============================================================================="