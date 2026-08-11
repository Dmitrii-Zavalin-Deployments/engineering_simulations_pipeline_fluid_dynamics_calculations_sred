#!/usr/bin/env bash
# ==============================================================================
# src/debug/forensic_audit.sh
# Forensic diagnostic and automated remediation script for integration test CLI arguments.
# ==============================================================================

set -euo pipefail

echo "============================================================"
echo "      [FORENSIC AUDIT] Integration Test CLI Diagnostics     "
echo "============================================================"

# echo "--- [1] Reproducing integration test failures ---"
# pytest tests/integration/test_01_main_ingestion.py tests/integration/test_02_main_state.py || true

echo ""
echo "--- [2] Inspecting integration test invocation in tests/integration/test_01_main_ingestion.py ---"
if [ -f "tests/integration/test_01_main_ingestion.py" ]; then
    grep -n -C 5 -- "input_output_folder" tests/integration/test_01_main_ingestion.py || true
else
    echo "Integration test file not found."
fi

echo ""
echo "--- [3] Inspecting argument validation in src/main.py ---"
if [ -f "src/main.py" ]; then
    grep -n -C 5 -- "--output_file_name" src/main.py || true
    echo ""
    echo "Source smoking-gun lines in src/main.py (100-125):"
    cat -n src/main.py | sed -n '100,125p'
else
    echo "CRITICAL: src/main.py does not exist."
    exit 1
fi

echo ""
echo "--- [4] Automated Repair Candidates (Reference # sed commands) ---"
echo "To update integration tests to include --output_file_name under non-default policy:"
# sed -i 's/"--input_output_folder", str(tmp_path)/"--input_output_folder", str(tmp_path), "--output_file_name", "simulation_results.zip"/g' tests/integration/test_01_main_ingestion.py
# sed -i 's/"--input_output_folder", str(tmp_path)/"--input_output_folder", str(tmp_path), "--output_file_name", "simulation_results.zip"/g' tests/integration/test_02_main_state.py

echo "============================================================"
echo "          [FORENSIC AUDIT] Diagnostic Suite Complete        "
echo "============================================================"