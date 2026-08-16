#!/usr/bin/env bash
# ==============================================================================
# src/debug/forensic_audit.sh
# Forensic Audit Script for Schema Validation Failures (Float vs Integer Mismatch)
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "[PHASE 1] Diagnostics: Locating root cause of float '0.2' in integer schema fields"
echo "=============================================================================="

echo "--- Searching for references to '0.2' across codebase ---"
grep -rn "0.2" src/ tests/ || true

echo "--- Searching for 'output_interval' and integer parameter definitions ---"
grep -rn "output_interval" src/ tests/ schema/ || true

echo "--- Inspecting generated output JSON payload for float/integer mismatches ---"
if [ -f "data/testing-input-output/navier_stokes_output.json" ]; then
    grep -rn "0.2" data/testing-input-output/navier_stokes_output.json || true
    cat data/testing-input-output/navier_stokes_output.json
fi

echo "=============================================================================="
echo "[PHASE 2] Smoking-Gun Source Audits (cat -n)"
echo "=============================================================================="

if [ -f "src/archivist.py" ]; then
    echo "--- Auditing src/archivist.py ---"
    cat -n src/archivist.py
fi

if [ -f "src/ingestion.py" ]; then
    echo "--- Auditing src/ingestion.py ---"
    cat -n src/ingestion.py
fi

if [ -f "src/state.py" ]; then
    echo "--- Auditing src/state.py ---"
    cat -n src/state.py
fi

echo "=============================================================================="
echo "[PHASE 3] Automated Repair Templates (sed injections)"
echo "=============================================================================="
echo "Use the following commented sed patterns for automated repairs:"
echo ""
echo "# sed -i 's/int(/int(float(/g' src/archivist.py"
echo "# sed -i 's/\"output_interval\": 0.2/\"output_interval\": 1/g' data/testing-input-output/navier_stokes_output.json"
echo "# sed -i 's/\"type\": \"integer\"/\"type\": \"number\"/g' schema/solver_input_schema.json"
echo "# sed -i 's/\"type\": \"integer\"/\"type\": \"number\"/g' schema/solver_output_schema.json"
echo ""
echo "Forensic audit execution completed successfully."