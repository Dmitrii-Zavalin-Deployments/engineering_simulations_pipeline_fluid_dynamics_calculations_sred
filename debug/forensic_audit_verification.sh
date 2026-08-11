#!/usr/bin/env bash
# ==============================================================================
# src/debug/forensic_audit.sh
# Forensic diagnostic and automated remediation script for integration test 
# SystemExit(1) failures and pipeline exception tracing.
# ==============================================================================

set -euo pipefail

echo "============================================================"
echo "    [FORENSIC AUDIT] Integration Test Exit Code 1 Diagnostics"
echo "============================================================"

echo "--- [1] Running pytest with full traceback on failing integration tests ---"
pytest tests/integration/test_01_main_ingestion.py tests/integration/test_02_main_state.py --tb=long || true

echo ""
echo "--- [2] Inspecting exception handling block in src/main.py ---"
if [ -f "src/main.py" ]; then
    grep -n -C 8 "except (" src/main.py || true
    echo ""
    echo "Source smoking-gun lines in src/main.py (120-145):"
    cat -n src/main.py | sed -n '120,145p'
else
    echo "CRITICAL: src/main.py does not exist."
    exit 1
fi

echo ""
echo "--- [3] Automated Repair Candidates (Reference # sed commands) ---"
echo "To print detailed exception class diagnostics before exit in main():"
# sed -i 's/print(f"FATAL PIPELINE ERROR: {e!s}", file=sys.stderr)/print(f"FATAL PIPELINE ERROR [{type(e).__name__}]: {e!s}", file=sys.stderr)/g' src/main.py

echo "============================================================"
echo "          [FORENSIC AUDIT] Diagnostic Suite Complete        "
echo "============================================================"