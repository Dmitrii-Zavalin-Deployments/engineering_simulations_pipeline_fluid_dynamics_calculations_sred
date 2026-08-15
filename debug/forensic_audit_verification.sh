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
    echo "--- src/main.py (Archivist Invocation Window) ---"
    grep -n -C 5 "archive_simulation_results" src/main.py || true
fi

echo -e "\n=========================================================================="
echo " [4/4] AUTOMATED REPAIR TEMPLATES (SED INJECTIONS)"
echo "=========================================================================="
echo "Uncomment the matching sed patch below based on the audited root cause:"

# Repair A: Fix return statement in archivist.py to return zip file path instead of json manifest path
# sed -i 's/return str(json_path)/return str(zip_file_path)/g' src/archivist.py

# Repair B: Ensure test expects the correct zip filename or structure
# sed -i 's/assert "output_summary.json" in namelist/assert "field_u.npy" in namelist/g' tests/integration/test_04_main_archivist.py

echo "Forensic audit run complete."