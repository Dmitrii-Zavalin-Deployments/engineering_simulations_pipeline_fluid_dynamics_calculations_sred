#!/bin/bash
# Description: Automated forensic audit for Navier-Stokes solver build/test failures.
# Triggered on CI failure to diagnose missing test directories and glob expansion issues.

echo "============================================================"
echo "🔍 FORENSIC AUDIT STARTING"
echo "============================================================"

echo "--- [1] Checking directory tree under cpp/ ---"
ls -R cpp/ || echo "❌ cpp directory structure missing or inaccessible."

echo "--- [2] Inspecting git status and untracked/missing files ---"
git status --porcelain

echo "--- [3] Smoking-Gun Source Audit: Checking test layout with cat -n ---"
if [ -d "cpp/tests" ]; then
    echo "Listing all files in cpp/tests recursively:"
    find cpp/tests -type f
    
    # If a specific test file exists, inspect its structure
    if [ -f "cpp/tests/test_advection.cpp" ]; then
        echo "Inspecting cpp/tests/test_advection.cpp header:"
        cat -n cpp/tests/test_advection.cpp | head -n 20
    fi
else
    echo "❌ Critical: cpp/tests directory does not exist!"
fi

echo "--- [4] Grep diagnostics for workflow compilation steps ---"
if [ -d ".github/workflows" ]; then
    grep -rn "cpp/tests" .github/workflows/ || echo "No direct workflow match found."
fi

echo "============================================================"
echo "🛠️ RECOMMENDED AUTOMATED REPAIRS (SED INJECTIONS):"
echo "============================================================"
# If wildcard expansion fails due to static paths, replace them with safe dynamic find commands:
# sed -i 's|cpp/tests/unit/\*.cpp|\$(find cpp/tests/unit -name "*.cpp")|g' .github/workflows/*.yml
# sed -i 's|cpp/tests/integration/\*.cpp|\$(find cpp/tests/integration -name "*.cpp")|g' .github/workflows/*.yml

# If test files are still flat in cpp/tests/ and need to be auto-migrated into subfolders:
# mkdir -p cpp/tests/unit cpp/tests/integration && mv cpp/tests/test_*.cpp cpp/tests/unit/ || true

echo "============================================================"