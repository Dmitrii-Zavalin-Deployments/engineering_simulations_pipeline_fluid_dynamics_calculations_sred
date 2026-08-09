#!/bin/bash
# Description: Automated forensic audit for Navier-Stokes solver build/test failures.
# Targets scope and namespace resolution issues for the Orchestrator class.

echo "============================================================"
echo "🔍 FORENSIC AUDIT: Orchestrator Scope & Namespace Diagnostics"
echo "============================================================"

echo "--- [1] Inspecting orchestrator.hpp for class definition and namespaces ---"
if [ -f "cpp/include/orchestrator.hpp" ]; then
    echo "Searching for Orchestrator declaration in headers:"
    grep -rn "class Orchestrator" cpp/include/ || grep -rn "Orchestrator" cpp/include/
else
    echo "❌ Critical: cpp/include/orchestrator.hpp not found."
fi

echo "--- [2] Smoking-Gun Source Audit: test_projection_pipeline.cpp ---"
if [ -f "cpp/cpp_integration_tests/test_projection_pipeline.cpp" ]; then
    echo "Inspecting lines 40 to 70 of test_projection_pipeline.cpp using cat -n:"
    cat -n cpp/cpp_integration_tests/test_projection_pipeline.cpp | sed -n '40,70p'
else
    echo "❌ Critical: test_projection_pipeline.cpp not found."
fi

echo "--- [3] Checking compiler include paths and file existence ---"
ls -la cpp/include/
ls -la cpp/cpp_integration_tests/

echo "============================================================"
echo "🛠️ RECOMMENDED AUTOMATED REPAIRS (SED INJECTIONS):"
echo "============================================================"
# If Orchestrator is wrapped inside a namespace (e.g., namespace navier_stokes), inject a using directive:
# sed -i '/#include "orchestrator.hpp"/a using namespace navier_stokes;' cpp/cpp_integration_tests/test_projection_pipeline.cpp

# Alternatively, qualify the class name directly across the test file if a namespace is mandatory:
# sed -i 's/\bOrchestrator\b/navier_stokes::Orchestrator/g' cpp/cpp_integration_tests/test_projection_pipeline.cpp

echo "============================================================"
echo "❌ Forensic audit complete. Exiting with failure status."
exit 1