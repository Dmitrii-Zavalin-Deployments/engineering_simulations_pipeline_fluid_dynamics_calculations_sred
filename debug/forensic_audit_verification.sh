#!/bin/bash
# Description: Automated forensic audit for Navier-Stokes solver advection sign inversion and gradient geometry validation failures.
# Status: Active (Triggered by Test Failures in Advection and Gradient suites)

echo "=== 🔍 FORENSIC AUDIT: ADVECTION SIGN INVERSION & GRADIENT GEOMETRY VALIDATION ==="

echo "--- 1. Searching for failure markers in test execution log ---"
if [ -f "build/tests/unit_tests" ]; then
    echo "Running unit tests under selective filtering to capture failure traces..."
    ./build/tests/unit_tests --gtest_filter=*AdvectionTest*:*GradientTest* || true
fi

echo "--- 2. Smoking-gun source audit: Locating Advection implementation ---"
ADV_IMPL=$(find cpp/src -name "advection.cpp" | head -n 1)
if [ -n "$ADV_IMPL" ]; then
    echo "Found Advection implementation: $ADV_IMPL"
    echo "Inspecting implementation via cat -n:"
    cat -n "$ADV_IMPL"
else
    echo "⚠️ Advection implementation file not found under cpp/src/."
fi

echo "--- 3. Smoking-gun source audit: Locating Gradient implementation ---"
GRAD_IMPL=$(find cpp/src -name "gradient.cpp" | head -n 1)
if [ -n "$GRAD_IMPL" ]; then
    echo "Found Gradient implementation: $GRAD_IMPL"
    echo "Inspecting implementation via cat -n:"
    cat -n "$GRAD_IMPL"
else
    echo "⚠️ Gradient implementation file not found under cpp/src/."
fi

echo "--- 4. Automated Repair Guidance (Sed Injections) ---"
echo "To fix advection sign inversion (getting -20 instead of 20) and add geometry validation for gradients:"
# sed -i 's/double advection_val = -(ui \* dfield_dx + vi \* dfield_dy + wi \* dfield_dz);/double advection_val = (ui * dfield_dx + vi * dfield_dy + wi * dfield_dz);/g' cpp/src/ops/advection.cpp
# sed -i 's/if (dx == 0.0 || dy == 0.0 || dz == 0.0)/if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0)/g' cpp/src/ops/gradient.cpp

