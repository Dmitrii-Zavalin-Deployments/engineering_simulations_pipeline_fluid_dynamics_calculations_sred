#!/usr/bin/env bash
set -euo pipefail

echo "================================================================="
echo "1. DIAGNOSTICS: Root Cause Analysis for Remaining 11 Failures"
echo "================================================================="

echo "--> Check 1: Remaining references to 'main_solver' in test files:"
grep -rn "main_solver" tests/ || echo "No 'main_solver' references found."

echo -e "\n--> Check 2: CLI exit code & exception handling in src/main.py:"
grep -A 20 "def main()" src/main.py || echo "src/main.py not found."

echo -e "\n--> Check 3: Schema contract output dummy structure in tests/property_integrity/test_schema_contracts.py:"
grep -A 30 "test_output_dummy_matches_schema" tests/property_integrity/test_schema_contracts.py 2>/dev/null || echo "Test not found."

echo "================================================================="
echo "2. SMOKING-GUN SOURCE AUDITS (cat -n)"
echo "================================================================="

echo "--> Auditing src/main.py main() entrypoint:"
if [ -f "src/main.py" ]; then
    tail -n 45 src/main.py | cat -n
fi

echo -e "\n--> Auditing tests/main_solver/test_cli_entry.py:"
if [ -f "tests/main_solver/test_cli_entry.py" ]; then
    cat -n tests/main_solver/test_cli_entry.py
fi

echo -e "\n--> Auditing tests/main_solver/test_guards.py:"
if [ -f "tests/main_solver/test_guards.py" ]; then
    cat -n tests/main_solver/test_guards.py
fi

echo -e "\n--> Auditing tests/property_integrity/test_schema_contracts.py:"
if [ -f "tests/property_integrity/test_schema_contracts.py" ]; then
    head -n 40 tests/property_integrity/test_schema_contracts.py | cat -n
fi

echo "================================================================="
echo "3. ROOT CAUSES & AUTOMATED REPAIRS"
echo "================================================================="