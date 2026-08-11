#!/usr/bin/env bash
# ==============================================================================
# src/debug/forensic_audit.sh
# Forensic diagnostic and automated remediation script for integration test 
# ValueError: too many values to unpack root causes.
# ==============================================================================

set -euo pipefail

echo "============================================================"
echo "    [FORENSIC AUDIT] Unpacking ValueError Diagnostics"
echo "============================================================"

# echo "--- [1] Running pytest on test_01_main_ingestion.py with full traceback ---"
# pytest tests/integration/test_01_main_ingestion.py --tb=long || true

echo ""
echo "--- [2] Inspecting test_01_main_ingestion.py assertion lines ---"
if [ -f "tests/integration/test_01_main_ingestion.py" ]; then
    grep -n -C 5 "call_args" tests/integration/test_01_main_ingestion.py || true
    echo ""
    echo "Source smoking-gun lines in tests/integration/test_01_main_ingestion.py (30-45):"
    cat -n tests/integration/test_01_main_ingestion.py | sed -n '30,45p'
else
    echo "CRITICAL: tests/integration/test_01_main_ingestion.py not found."
    exit 1
fi

echo ""
echo "--- [3] Automated Repair Candidates (Reference # sed commands) ---"
echo "To capture the return value correctly from a wrapped mock function:"
# sed -i 's/input_data, config_data = spy_ingest.call_args.result/input_data, config_data = spy_ingest.return_value/g' tests/integration/test_01_main_ingestion.py

echo "============================================================"
echo "          [FORENSIC AUDIT] Diagnostic Suite Complete        "
echo "============================================================"