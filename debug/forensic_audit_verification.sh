#!/usr/bin/env bash
# ============================================================================
# File: src/debug/forensic_audit.sh
# Description: Forensic probe to determine if NavierStokesSolver(None) 
#              raises a clean exception or causes a low-level C++ crash.
# ============================================================================
set -euo pipefail

echo "=== [FORENSIC AUDIT] Probing NavierStokesSolver(None) behavior ==="

PYTHONPATH="$PWD:$PWD/build" LD_PRELOAD="$(g++ -print-file-name=libasan.so)" python3 -c '
import sys
try:
    import navier_stokes_cpp
    print("Module loaded successfully.")
    navier_stokes_cpp.NavierStokesSolver(None)
    print("ERROR: NavierStokesSolver(None) returned without error!")
except Exception as e:
    print(f"Caught Python Exception: {type(e).__name__}: {e}")
    sys.exit(0)
except BaseException as e:
    print(f"Caught BaseException: {type(e).__name__}: {e}")
    sys.exit(1)
'

echo "=== [FORENSIC AUDIT] Probe completed. ==="