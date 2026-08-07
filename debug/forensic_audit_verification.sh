#!/usr/bin/env bash
set -euo pipefail

echo "======================================================================"
echo "🔍 STARTING FORENSIC AUDIT: C++ Compilation Failure Diagnosis"
echo "======================================================================"

echo ""
echo "--- 1. DIAGNOSTICS: Root Cause Identification ---"
echo "Checking for pybind11 installation and headers across standard paths:"
find /usr -name "pybind11.h" 2>/dev/null || echo "⚠️ Warning: pybind11.h not found in system include paths. Check Python bindings setup."

echo ""
echo "Scanning test files for namespace resolution indicators:"
grep -rn "TEST(" cpp/tests/ --include="*.cpp" -A 2 || true

echo ""
echo "--- 2. SMOKING-GUN SOURCE AUDITS (cat -n) ---"
echo "Inspecting top lines of cpp/tests/test_advection.cpp for missing namespace declarations:"
cat -n cpp/tests/test_advection.cpp | head -n 30

echo ""
echo "Inspecting header encapsulation in cpp/include/advection.hpp:"
cat -n cpp/include/advection.hpp | head -n 25

echo ""
echo "--- 3. AUTOMATED REPAIRS (sed injections) ---"
echo "Note: The following sed commands can be uncommented to automatically inject"
echo "'using namespace ops;' into test suites where operators are called without scope resolution."
echo ""

# sed -i '/#include <gtest\/gtest.h>/a using namespace ops;' cpp/tests/test_advection.cpp
# sed -i '/#include <gtest\/gtest.h>/a using namespace ops;' cpp/tests/test_divergence.cpp
# sed -i '/#include <gtest\/gtest.h>/a using namespace ops;' cpp/tests/test_forces.cpp
# sed -i '/#include <gtest\/gtest.h>/a using namespace ops;' cpp/tests/test_ghost_handler.cpp
# sed -i '/#include <gtest\/gtest.h>/a using namespace ops;' cpp/tests/test_gradient.cpp
# sed -i '/#include <gtest\/gtest.h>/a using namespace ops;' cpp/tests/test_laplacian.cpp
# sed -i '/#include <gtest\/gtest.h>/a using namespace ops;' cpp/tests/test_scaling.cpp

echo "======================================================================"
echo "🏁 Forensic Audit Script Complete."
echo "======================================================================"