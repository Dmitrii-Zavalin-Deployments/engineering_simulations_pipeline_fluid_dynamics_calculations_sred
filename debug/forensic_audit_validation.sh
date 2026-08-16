#!/usr/bin/env bash
# ==============================================================================
# src/debug/forensic_audit.sh
# Forensic Audit Script for Schema Validation Failures (Float vs Integer Mismatch)
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "[PHASE 1] Diagnostics: Locating schema files, validation logic, and float values"
echo "=============================================================================="

echo "--- Searching for JSON schema definition files ---"
find . -name "*.json" -not -path "*/.*" -not -path "*/venv/*" || true

echo "--- Searching for schema validation execution points ---"
grep -rn "validate" src/ tests/ || true
grep -rn "schema" src/ tests/ || true

echo "--- Searching for references to '0.2' in codebase ---"
grep -rn "0.2" src/ tests/ || true

echo "=============================================================================="
echo "[PHASE 2] Smoking-Gun Source Audits (cat -n)"
echo "=============================================================================="

if [ -f "src/utils/validate_schema.py" ]; then
    echo "--- Auditing src/utils/validate_schema.py ---"
    cat -n src/utils/validate_schema.py
fi

# Locate and audit any schema json files found in the repository
for schema_file in $(find . -name "*schema*.json" -not -path "*/venv/*" 2>/dev/null); do
    echo "--- Auditing schema file: $schema_file ---"
    cat -n "$schema_file"
done

echo "=============================================================================="
echo "[PHASE 3] Automated Repair Templates (sed injections)"
echo "=============================================================================="
echo "Use the following commented sed patterns to adjust schema types or parsing logic:"
echo ""
echo "# sed -i 's/\"type\": \"integer\"/\"type\": \"number\"/g' path/to/schema.json"
echo "# sed -i 's/\"type\":\s*\"integer\"/\"type\": \"number\"/g' src/utils/validate_schema.py"
echo "# sed -i 's/int(/float(/g' src/utils/validate_schema.py"
echo ""
echo "Forensic audit execution completed successfully."