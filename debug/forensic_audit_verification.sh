#!/usr/bin/env bash
# ==============================================================================
# src/debug/forensic_audit.sh
# GitHub Actions Post-Test Forensic Audit & Repair Script for BadZipFile Failures
# ==============================================================================

set -uo pipefail

echo "=========================================================================="
echo " [1/4] FORENSIC DIAGNOSTICS: ZIP ARTIFACT & LOG AUDIT"
echo "=========================================================================="

find . -maxdepth 4 \( -name "*.zip" -o -name "*.tmp" -o -name "*simulation*" \) -not -path "*/.*" | while read -r filepath; do
    if [ -f "$filepath" ]; then
        echo "--- File Info: $filepath ---"
        ls -la "$filepath"
        file "$filepath" || true
        echo "Header Bytes (Hex/ASCII dump):"
        head -c 128 "$filepath" | xxd 2>/dev/null || head -c 128 "$filepath" || true
        echo -e "\n--------------------------------------------------"
    fi
done

echo -e "\n=========================================================================="
echo " [2/4] GREP DIAGNOSTICS: ARCHIVIST & ZIP CODE PATHS"
echo "=========================================================================="

echo "--- Searching for archivist calls and zipfile operations ---"
grep -rn "archive_simulation_results" src/ tests/ || true
grep -rn "zipfile" src/ tests/ || true
grep -rn "BadZipFile" src/ tests/ || true

echo -e "\n=========================================================================="
echo " [3/4] SMOKING-GUN SOURCE AUDITS (cat -n)"
echo "=========================================================================="

if [ -f "src/archivist.py" ]; then
    echo "--- src/archivist.py ---"
    cat -n src/archivist.py
fi

if [ -f "tests/integration/test_04_main_archivist.py" ]; then
    echo "--- tests/integration/test_04_main_archivist.py ---"
    cat -n tests/integration/test_04_main_archivist.py
fi

if [ -f "src/main.py" ]; then
    echo "--- src/main.py ---"
    cat -n src/main.py
fi

echo -e "\n=========================================================================="
echo " [4/4] AUTOMATED REPAIR TEMPLATES (SED INJECTIONS)"
echo "=========================================================================="
echo "Applying automated fix for positional argument mismatch in src/main.py..."

# Fix positional argument mapping where zip_filename was passed as output_json_filename
# sed -i 's/archive_simulation_results(state, str(out_path), zip_filename)/archive_simulation_results(state, str(out_path), zip_filename=zip_filename)/g' src/main.py

echo "Forensic audit and repair sequence complete."