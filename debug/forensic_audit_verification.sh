#!/usr/bin/env bash
# src/debug/forensic_audit.sh
# Forensic diagnostic and automated repair script for CTest/GoogleTest stack smashing crashes.

set -euo pipefail

echo "================================================================="
echo " FORENSIC AUDIT: STACK SMASHING & BUFFER OVERFLOW DIAGNOSTICS"
echo "================================================================="

# -----------------------------------------------------------------------------
# 1. DIAGNOSTICS: GREP & ROOT CAUSE SEARCH
# -----------------------------------------------------------------------------
echo -e "\n[+] Audit Step 1: Searching for stack-allocated raw buffers (primary cause of stack smashing)..."
grep -rnE "double\s+[a-zA-Z0-9_]+\s*\[" tests/ cpp/ tests/ 2>/dev/null || true
grep -rnE "float\s+[a-zA-Z0-9_]+\s*\[" tests/ cpp/ tests/ 2>/dev/null || true
grep -rnE "int\s+[a-zA-Z0-9_]+\s*\[" tests/ cpp/ tests/ 2>/dev/null || true

echo -e "\n[+] Audit Step 2: Locating MassContinuity test implementation files..."
MASS_TEST_FILE=$(find . -type f \( -name "*mass_continuity*.cpp" -o -name "*MassContinuity*.cpp" \) | head -n 1)

if [ -z "${MASS_TEST_FILE}" ]; then
    echo "[-] WARNING: MassContinuity test file not found in standard paths. Searching all test files..."
    MASS_TEST_FILE=$(grep -rl "EnforcesZeroDivergenceInFluidDomain" . | head -n 1)
fi

echo "--> Target Test File: ${MASS_TEST_FILE}"

echo -e "\n[+] Audit Step 3: Checking for grid dimension arithmetic vs allocation size mismatches..."
grep -rnC 5 "EnforcesZeroDivergenceInFluidDomain" . 2>/dev/null || true

# -----------------------------------------------------------------------------
# 2. SMOKING-GUN SOURCE AUDITS (cat -n)
# -----------------------------------------------------------------------------
echo -e "\n[+] Audit Step 4: Printing source code with line numbers for smoking-gun analysis..."
if [ -n "${MASS_TEST_FILE}" ] && [ -f "${MASS_TEST_FILE}" ]; then
    echo "================================================================="
    echo " SOURCE CODE: ${MASS_TEST_FILE}"
    echo "================================================================="
    cat -n "${MASS_TEST_FILE}"
else
    echo "[-] ERROR: Could not locate MassContinuity test source for printing."
fi

echo -e "\n[+] Audit Step 5: Printing grid math flat index helper header for bounds verification..."
GRID_MATH_HEADER=$(find . -type f -name "grid_math.hpp" | head -n 1)
if [ -n "${GRID_MATH_HEADER}" ] && [ -f "${GRID_MATH_HEADER}" ]; then
    echo "================================================================="
    echo " SOURCE CODE: ${GRID_MATH_HEADER}"
    echo "================================================================="
    cat -n "${GRID_MATH_HEADER}"
fi

# -----------------------------------------------------------------------------
# 3. SED INJECTIONS FOR AUTOMATED REPAIRS (# prepended to each sed command)
# -----------------------------------------------------------------------------
echo -e "\n[+] Audit Step 6: Prepared Automated Repair Commands (Uncomment in script to apply)"

# Repair Strategy A: Convert stack-allocated double arrays to dynamic std::vector allocations
# # sed -i 's/double u\[\(.*\)\];/std::vector<double> u(\1, 0.0);/g' "${MASS_TEST_FILE}"
# # sed -i 's/double v\[\(.*\)\];/std::vector<double> v(\1, 0.0);/g' "${MASS_TEST_FILE}"
# # sed -i 's/double w\[\(.*\)\];/std::vector<double> w(\1, 0.0);/g' "${MASS_TEST_FILE}"
# # sed -i 's/double p\[\(.*\)\];/std::vector<double> p(\1, 0.0);/g' "${MASS_TEST_FILE}"
# # sed -i 's/double div\[\(.*\)\];/std::vector<double> div(\1, 0.0);/g' "${MASS_TEST_FILE}"
# # sed -i 's/double div_out\[\(.*\)\];/std::vector<double> div_out(\1, 0.0);/g' "${MASS_TEST_FILE}"

# Repair Strategy B: Update operator function call signatures to pass .data() pointers for std::vector
# # sed -i 's/compute_divergence(u, v, w, div/compute_divergence(u.data(), v.data(), w.data(), div.data()/g' "${MASS_TEST_FILE}"
# # sed -i 's/compute_divergence(u_star, v_star, w_star, div_out/compute_divergence(u_star.data(), v_star.data(), w_star.data(), div_out.data()/g' "${MASS_TEST_FILE}"

# Repair Strategy C: Fix 2D vs 3D cell count allocation bugs (e.g., Nx * Ny -> Nx * Ny * Nz)
# # sed -i 's/size_t total_cells = Nx \* Ny;/size_t total_cells = static_cast<size_t>(Nx) \* Ny \* Nz;/g' "${MASS_TEST_FILE}"
# # sed -i 's/int total_cells = Nx \* Ny;/int total_cells = Nx \* Ny \* Nz;/g' "${MASS_TEST_FILE}"

echo -e "\n================================================================="
echo " FORENSIC AUDIT SCRIPT COMPLETE"
echo "================================================================="