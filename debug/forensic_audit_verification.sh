#!/usr/bin/env bash
# ==============================================================================
# @file forensic_audit.sh
# @brief Post-test diagnostic audit for schema validation and grid dimension failures
# ==============================================================================

set -uo pipefail

echo "=============================================================================="
echo "                   FORENSIC AUDIT: SCHEMA VALIDATION FAILURES               "
echo "=============================================================================="

echo ""
echo "=== [1/3] Diagnostic Grep & Output Analysis ==="
echo "Searching for grid minimum constraints and schema path references..."
grep -rn "minimum" schemas/ src/ 2>/dev/null || echo "No explicit 'minimum' found via grep in schemas/ or src/"
grep -rn "nz" schemas/ src/ 2>/dev/null || true

echo ""
echo "=== [2/3] Smoking-Gun Source Audits (cat -n) ==="
echo "Inspecting schema validation logic and ingestion module (covering lines 50-95 in ingestion.py):"

if [ -f "src/ingestion.py" ]; then
    echo "--- src/ingestion.py (Targeting uncovered lines: 60, 62, 79-80, 85, 89) ---"
    sed -n '50,95p' src/ingestion.py | cat -n
else
    echo "Warning: src/ingestion.py not found in current directory."
fi

if [ -f "src/utils/validate_schema.py" ]; then
    echo "--- src/utils/validate_schema.py ---"
    cat -n src/utils/validate_schema.py
fi

# Search for any json schema files in the workspace
SCHEMA_FILES=$(find . -name "*.json" -not -path "*/.*" -not -path "*/venv/*" 2>/dev/null)
for sf in $SCHEMA_FILES; do
    if grep -q "grid" "$sf" 2>/dev/null; then
        echo "--- Found Schema Definition File: $sf ---"
        cat -n "$sf"
    fi
done

echo ""
echo "=== [3/3] Automated Repair Options (Sed Injections) ==="
echo "The following commented sed commands can be uncommented to dynamically adjust"
echo "the minimum grid constraints (e.g., lowering minimum dimension requirement from 4 to 2"
echo "to support non-cubic test grids like 5x4x3):"
echo ""
# sed -i 's/"minimum": 4/"minimum": 2/g' schemas/*.json 2>/dev/null || true
# sed -i 's/minItems": 4/minItems": 2/g' schemas/*.json 2>/dev/null || true
# sed -i 's/min_grid_size = 4/min_grid_size = 2/g' src/utils/validate_schema.py 2>/dev/null || true

echo "=============================================================================="
echo "                        FORENSIC AUDIT COMPLETE                               "
echo "=============================================================================="