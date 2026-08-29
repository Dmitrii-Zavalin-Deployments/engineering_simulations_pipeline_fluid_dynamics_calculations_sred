#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="./build"
TEST_BIN="${BUILD_DIR}/test_full_pipeline_accelerated_flow"
FILTER="FullPipelineAcceleratedFlowTest.StepByStepAccelerated"
LOG_DIR="./solver_debug_logs"

mkdir -p "${LOG_DIR}"

echo "===================================================="
echo " 1. RUNNING TEST IN ISOLATION (Single Test Run)"
echo "===================================================="
"${TEST_BIN}" --gtest_filter="${FILTER}" > "${LOG_DIR}/01_solo_run.log" 2>&1 || true

echo "===================================================="
echo " 2. TESTING SINGLE-THREADED EXECUTION (Isolate Race Conditions)"
echo "===================================================="
OMP_NUM_THREADS=1 "${TEST_BIN}" --gtest_filter="${FILTER}" > "${LOG_DIR}/02_single_threaded.log" 2>&1 || true

echo "===================================================="
echo " 3. TESTING STATE LEAKAGE (Run full suite vs. Solo)"
echo "===================================================="
# Run all tests in binary to see if preceding tests contaminate state
"${TEST_BIN}" > "${LOG_DIR}/03_full_suite_run.log" 2>&1 || true

echo "===================================================="
echo " 4. CHECKING UNINITIALIZED MEMORY / LEAKS (Valgrind)"
echo "===================================================="
if command -v valgrind &> /dev/null; then
    valgrind --tool=memcheck --leak-check=full --track-origins=yes \
        "${TEST_BIN}" --gtest_filter="${FILTER}" > "${LOG_DIR}/04_valgrind.log" 2>&1 || true
    echo "Valgrind complete. Check ${LOG_DIR}/04_valgrind.log for uninitialized reads."
else
    echo "Valgrind not found, skipping."
fi

echo "===================================================="
echo " DIAGNOSTIC SUMMARY & DIFF"
echo "===================================================="

echo "--- Solo Run vs Single-Threaded Diff ---"
diff -u <(grep "snap.u_star" "${LOG_DIR}/01_solo_run.log" || true) \
        <(grep "snap.u_star" "${LOG_DIR}/02_single_threaded.log" || true) || echo "Threading caused numeric changes."

echo ""
echo "--- Check for Uninitialized Memory Warnings ---"
grep -E "Conditional jump or move depends on uninitialised value|Uninitialised value" "${LOG_DIR}/04_valgrind.log" 2>/dev/null || echo "No obvious uninitialized reads in Valgrind."

echo ""
echo "All diagnostic logs saved to ${LOG_DIR}/"