#!/usr/bin/env bash
# ==============================================================================
# File: src/debug/forensic_audit.sh
# Purpose: Post-test forensic audit script for missing header errors 
#          (nlohmann/json.hpp) during C++ unit/integration test compilation.
# ==============================================================================
set -euo pipefail

echo "========================================================================"
echo "🔍 STAGE 1: SYSTEM ENVIRONMENT & HEADER DIAGNOSTICS"
echo "========================================================================"

echo "---> Searching for nlohmann/json.hpp across system directories:"
find /usr /opt /home/runner -name "json.hpp" 2>/dev/null || echo "⚠️ nlohmann/json.hpp not found in filesystem."

echo -e "\n---> Checking installed system packages for nlohmann-json:"
if command -v dpkg >/dev/null 2>&1; then
    dpkg -l | grep -i nlohmann || echo "⚠️ nlohmann-json package is not installed via apt."
fi

echo -e "\n---> Auditing active CI workflow files for build step triggers:"
grep -rn "Compiling and executing native" .github/workflows/ || echo "⚠️ Exact trigger string not found in .github/workflows/"

echo "========================================================================"
echo "📄 STAGE 2: SMOKING-GUN SOURCE AUDITS (cat -n)"
echo "========================================================================"

echo "---> Auditing Integration Test CMake configuration:"
if [ -f "cpp/cpp_integration_tests/CMakeLists.txt" ]; then
    cat -n cpp/cpp_integration_tests/CMakeLists.txt
else
    echo "⚠️ File cpp/cpp_integration_tests/CMakeLists.txt not found!"
fi

echo -e "\n---> Auditing test_projection_pipeline.cpp include headers (lines 35-55):"
if [ -f "cpp/cpp_integration_tests/test_projection_pipeline.cpp" ]; then
    sed -n '35,55p' cpp/cpp_integration_tests/test_projection_pipeline.cpp | cat -n
else
    echo "⚠️ File cpp/cpp_integration_tests/test_projection_pipeline.cpp not found!"
fi

echo "========================================================================"
echo "🔧 STAGE 3: AUTOMATED REPAIR SED INJECTIONS (Commented Recipes)"
echo "========================================================================"
echo "# Run or uncomment one of the following sed commands to repair the root cause:"

# ------------------------------------------------------------------------------
# REPAIR OPTION A: Add system package installation step to GitHub Actions workflow
# ------------------------------------------------------------------------------
# sed -i '/run: echo "🚀 Compiling/i \      - name: Install nlohmann-json3-dev\n        run: sudo apt-get update && sudo apt-get install -y nlohmann-json3-dev' .github/workflows/*.yml

# ------------------------------------------------------------------------------
# REPAIR OPTION B: Inject CMake FetchContent into CMakeLists.txt (Self-contained)
# ------------------------------------------------------------------------------
# sed -i '/find_package(GTest REQUIRED)/i \
# include(FetchContent)\n\
# FetchContent_Declare(\n\
#     nlohmann_json\n\
#     URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz\n\
# )\n\
# FetchContent_MakeAvailable(nlohmann_json)\n' cpp/cpp_integration_tests/CMakeLists.txt

# ------------------------------------------------------------------------------
# REPAIR OPTION C: Fallback to single-header curl download directly in build step
# ------------------------------------------------------------------------------
# sed -i '/run: echo "🚀 Compiling/i \      - name: Fetch nlohmann/json.hpp\n        run: sudo mkdir -p /usr/include/nlohmann && sudo curl -sSL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -o /usr/include/nlohmann/json.hpp' .github/workflows/*.yml

echo -e "\nForensic audit script execution complete."
exit 1