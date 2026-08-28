#!/usr/bin/env bash
# ==============================================================================
# src/debug/forensic_audit.sh
# Post-Test Forensic Audit & Diagnostic Script for GitHub Actions
# ==============================================================================

set -euo pipefail

echo "=== [1] Log Diagnostics & Root Cause Analysis ==="
if [ -f "test_detail.log" ]; then
    grep -n -C 5 "ASSERT_" test_detail.log || true
    grep -n -C 5 "EXPECT_" test_detail.log || true
    grep -n -C 5 "FAILED" test_detail.log || true
elif [ -f "build/Testing/Temporary/LastTest.log" ]; then
    echo "Found CTest LastTest.log. Scanning for test failures and assertions:"
    grep -n -C 5 "FAIL" build/Testing/Temporary/LastTest.log || true
    grep -n -C 5 "Error" build/Testing/Temporary/LastTest.log || true
else
    echo "No log files found. Capturing recent test output via ctest search:"
    if command -v ctest &> /dev/null; then
        cd build && ctest --output-on-failure --rerun-failed || true
    else
        find . -name "*.log" -o -name "LastTest.log" -exec grep -n -C 3 "FAIL\|ERROR\|ASSERT" {} + || true
    fi
fi

echo -e "\n=== [2] Smoking-Gun Source Audit (cat -n) ==="
TARGET_FILE="cpp/src/orchestrator.cpp"
if [ -f "$TARGET_FILE" ]; then
    echo "Inspecting source code around Rhie-Chow execution and boundary logic in $TARGET_FILE:"
    cat -n "$TARGET_FILE" | sed -n '130,195p'
else
    find cpp -maxdepth 3 -type f
fi

echo -e "\n=== [3] Automated Repair Staging (Sed Injections) ==="
# Uncomment to apply automated fixes if boundary index handling or face mapping requires adjustment:
# sed -i 's/const double u_east = /const double u_east = /g' cpp/src/orchestrator.cpp

echo "=== Forensic Audit Executed Successfully ==="