#!/usr/bin/env bash
# ==============================================================================
# src/debug/forensic_audit.sh
# Forensic diagnostic and automated remediation script for integration test 
# TypeError: cannot unpack non-iterable _SentinelObject object root causes.
# ==============================================================================

set -euo pipefail

echo "============================================================"
echo "    [FORENSIC AUDIT] Ingestion Test TypeError Diagnostics"
echo "============================================================"

# echo "--- [1] Running pytest on test_01_main_ingestion.py with full traceback ---"
# pytest tests/integration/test_01_main_ingestion.py --tb=long || true

echo ""
echo "--- [2] Inspecting test_01_main_ingestion.py mock return value handling ---"
if [ -f "tests/integration/test_01_main_ingestion.py" ]; then
    grep -n -C 5 "spy_ingest" tests/integration/test_01_main_ingestion.py || true
    echo ""
    echo "Source smoking-gun lines in tests/integration/test_01_main_ingestion.py (20-40):"
    cat -n tests/integration/test_01_main_ingestion.py | sed -n '20,40p'
else
    echo "CRITICAL: tests/integration/test_01_main_ingestion.py not found."
    exit 1
fi

echo ""
echo "--- [3] Automated Repair Candidates (Reference # sed commands) ---"
echo "To fix return value unpacking from spy_ingest via call result:"
# sed -i 's/input_data, config_data = spy_ingest.return_value/input_data, config_data = spy_ingest.call_args.result/g' tests/integration/test_01_main_ingestion.py

echo "============================================================"
echo "          [FORENSIC AUDIT] Diagnostic Suite Complete        "
echo "============================================================"