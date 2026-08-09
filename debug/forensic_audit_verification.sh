#!/usr/bin/env bash
# ==============================================================================
# File: src/debug/forensic_audit.sh
# Purpose: Post-test forensic audit script for missing data/config.json path 
#          issues during native C++ integration test execution.
# ==============================================================================
set -euo pipefail

echo "========================================================================"
echo "🔍 STAGE 1: FILE EXISTENCE & WORKING DIRECTORY DIAGNOSTICS"
echo "========================================================================"

echo "---> Searching for config.json across repository:"
find . -name "config.json" || echo "⚠️ config.json not found in repository."

echo -e "\n---> Checking current working directory and contents:"
pwd
ls -la

echo -e "\n---> Inspecting build/tests directory contents:"
ls -la build/tests/ || echo "⚠️ build/tests directory does not exist."

echo "========================================================================"
echo "📄 STAGE 2: SMOKING-GUN SOURCE AUDITS (cat -n)"
echo "========================================================================"

echo "---> Auditing test_projection_pipeline.cpp file opening lines (45-65):"
if [ -f "cpp/cpp_integration_tests/test_projection_pipeline.cpp" ]; then
    sed -n '45,65p' cpp/cpp_integration_tests/test_projection_pipeline.cpp | cat -n
else
    echo "⚠️ File cpp/cpp_integration_tests/test_projection_pipeline.cpp not found!"
fi

echo "========================================================================"
echo "🔧 STAGE 3: AUTOMATED REPAIR SED INJECTIONS (Commented Recipes)"
echo "========================================================================"
echo "# Run or uncomment one of the following sed commands to repair the root cause:"

# ------------------------------------------------------------------------------
# REPAIR OPTION A: Copy data folder into build/tests/ in workflow compilation step
# ------------------------------------------------------------------------------
# sed -i '/g++.*integration_tests/a \          cp -r cpp/cpp_integration_tests/data build/tests/' .github/workflows/navier_stokes_solver.yml

# ------------------------------------------------------------------------------
# REPAIR OPTION B: Update path in test_projection_pipeline.cpp to search cpp/cpp_integration_tests/data/
# ------------------------------------------------------------------------------
# sed -i 's|"data/config.json"|"cpp/cpp_integration_tests/data/config.json"|g' cpp/cpp_integration_tests/test_projection_pipeline.cpp
# sed -i 's|"data/navier_stokes_input.json"|"cpp/cpp_integration_tests/data/navier_stokes_input.json"|g' cpp/cpp_integration_tests/test_projection_pipeline.cpp

echo -e "\nForensic audit script execution complete."
