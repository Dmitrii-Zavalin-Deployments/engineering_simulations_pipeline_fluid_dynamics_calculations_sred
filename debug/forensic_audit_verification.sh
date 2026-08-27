#!/usr/bin/env bash
# ============================================================================
# File: src/debug/forensic_audit.sh
# Description: Post-test forensic audit script to isolate root-level module 
#              import failures and C++ initialization crashes.
# ============================================================================
set -euo pipefail

echo "=== [FORENSIC AUDIT] Starting Root Import & C++ Bridge Diagnostic ==="

# 1. Verify exact binary existence in the repository root
echo "--- Scanning root directory for compiled shared libraries ---"
ls -la navier_stokes_cpp*.so || echo "CRITICAL: No compiled .so file found in root directory."

# 2. Test direct Python import behavior and catch segmentation faults / tracebacks
echo "--- Testing isolated Python import of navier_stokes_cpp ---"
python3 -c "
import sys
print('Python Path:', sys.path)
try:
    import navier_stokes_cpp
    print('SUCCESS: navier_stokes_cpp imported successfully!')
    print('Module attributes:', dir(navier_stokes_cpp))
except Exception as e:
    print('IMPORT FAILED:', type(e).__name__, e)
    raise
"

# 3. Run pytest with verbose short tracebacks to capture collection errors
echo "--- Executing pytest with verbose collection diagnostics ---"
pytest --tb=short -v cpp/python_bridge_tests/ || true

echo "=== [FORENSIC AUDIT] Diagnostic audit completed. ==="