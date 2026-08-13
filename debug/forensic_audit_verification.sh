#!/bin/bash
# Description: Automated forensic audit for Navier-Stokes solver advection exceptions (Exit Code 134).
# Status: Active (Triggered by CI Failure SIGABRT / Uncaught Exception)

echo "=== 🔍 FORENSIC AUDIT: EXIT CODE 134 (SIGABRT / UNCAUGHT EXCEPTION IN ADVECTION) ==="

echo "--- 1. Searching for failure markers in test execution log ---"
if [ -f "build/tests/unit_tests" ]; then
    echo "Running unit tests under verbose inspection to capture stack trace..."
    ./build/tests/unit_tests --gtest_filter=*NonFiniteVelocityFieldThrows* || true
fi

echo "--- 2. Smoking-gun source audit: Locating test definition ---"
TEST_FILE=$(grep -rn "NonFiniteVelocityFieldThrows" cpp/cpp_unit_tests/ --include="*.cpp" | cut -d: -f1 | head -n 1)
if [ -n "$TEST_FILE" ]; then
    echo "Found test file: $TEST_FILE"
    LINE_NUM=$(grep -n "NonFiniteVelocityFieldThrows" "$TEST_FILE" | cut -d: -f1)
    echo "Inspecting context around line $LINE_NUM in $TEST_FILE:"
    cat -n "$TEST_FILE" | sed -n "$((LINE_NUM - 2)),$((LINE_NUM + 25))p"
else
    echo "⚠️ Test definition file not found for NonFiniteVelocityFieldThrows."
fi

echo "--- 3. Smoking-gun source audit: Locating exception trigger in implementation ---"
IMPL_FILE=$(grep -rn "Advection term exploded" cpp/src/ --include="*.cpp" | cut -d: -f1 | head -n 1)
if [ -n "$IMPL_FILE" ]; then
    echo "Found implementation file: $IMPL_FILE"
    IMPL_LINE=$(grep -n "Advection term exploded" "$IMPL_FILE" | cut -d: -f1)
    echo "Inspecting context around line $IMPL_LINE in $IMPL_FILE:"
    cat -n "$IMPL_FILE" | sed -n "$((IMPL_LINE - 8)),$((PL_LINE + 15))p"
else
    echo "⚠️ Implementation file containing 'Advection term exploded' not found."
fi

echo "--- 4. Automated Repair Guidance (Sed Injections) ---"
echo "To prevent cross-thread termination inside OpenMP parallel regions, ensure exceptions are thrown post-loop:"
# sed -i 's/throw std::runtime_error("Advection term exploded");/has_error = true;/g' cpp/src/ops/advection.cpp

