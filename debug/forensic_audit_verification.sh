#!/usr/bin/env bash
set -euo pipefail

echo "=================================================================="
echo "               FORENSIC AUDIT: GTEST FAILURE DIAGNOSTICS          "
echo "=================================================================="

# 1. Log Diagnostics for Test Execution Output
echo "[+] Step 1: Scanning test logs for assertion failures..."
if [ -f "logs.txt" ]; then
    grep -En "Failure|FAILED|mixing TEST_F and TEST|throws nothing" logs.txt || true
else
    echo "[-] logs.txt not found in working directory."
fi

echo ""
echo "[+] Step 2: Performing smoking-gun source audits (cat -n)..."

# 2. Audit PressurePoissonTest (TEST vs TEST_F macro mismatch)
POISSON_TEST_FILE=$(find cpp -type f -name "test_pressure_poisson.cpp" 2>/dev/null | head -n 1)
if [ -n "$POISSON_TEST_FILE" ] && [ -f "$POISSON_TEST_FILE" ]; then
    echo "--- Source Audit: $POISSON_TEST_FILE (Lines surrounding macro usage) ---"
    cat -n "$POISSON_TEST_FILE" | grep -C 5 "SolidNeumannBoundaryZeroNormalGradient" || true
else
    echo "[-] test_pressure_poisson.cpp not found."
fi

# 3. Audit PredictorTest Line 137 & Implementation Guard
PREDICTOR_TEST_FILE=$(find cpp -type f -name "test_predictor.cpp" 2>/dev/null | head -n 1)
if [ -n "$PREDICTOR_TEST_FILE" ] && [ -f "$PREDICTOR_TEST_FILE" ]; then
    echo "--- Source Audit: $PREDICTOR_TEST_FILE (Lines 130–145) ---"
    cat -n "$PREDICTOR_TEST_FILE" | sed -n '130,145p' || true
else
    echo "[-] test_predictor.cpp not found."
fi

PREDICTOR_IMPL_FILE=$(find cpp -type f -name "predictor.cpp" 2>/dev/null | head -n 1)
if [ -n "$PREDICTOR_IMPL_FILE" ] && [ -f "$PREDICTOR_IMPL_FILE" ]; then
    echo "--- Source Audit: $PREDICTOR_IMPL_FILE (Density validation check) ---"
    cat -n "$PREDICTOR_IMPL_FILE" | grep -C 5 "density" || true
else
    echo "[-] predictor.cpp implementation file not found."
fi

echo ""
echo "[+] Step 3: Automated Repair Blueprint (Commented out)"

# ------------------------------------------------------------------
# AUTOMATED REPAIRS (Uncomment individual commands as needed)
# ------------------------------------------------------------------

# Fix 1: Convert illegal TEST macro to TEST_F in PressurePoissonTest
# sed -i 's/TEST(PressurePoissonTest, SolidNeumannBoundaryZeroNormalGradient)/TEST_F(PressurePoissonTest, SolidNeumannBoundaryZeroNormalGradient)/g' "${POISSON_TEST_FILE}"

# Fix 2: Inject density validation check into compute_trial_velocities implementation if missing
# sed -i '/if (fluid\.nu < 0\.0)/i \    if (fluid.density <= 0.0) throw std::invalid_argument("Density must be strictly positive.");' "${PREDICTOR_IMPL_FILE}"