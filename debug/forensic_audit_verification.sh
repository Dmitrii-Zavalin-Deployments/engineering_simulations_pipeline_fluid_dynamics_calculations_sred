#!/bin/bash
set -eo pipefail

echo "=================================================================="
echo "      NAVIER-STOKES SOLVER: STACK SMASHING FORENSIC AUDIT        "
echo "=================================================================="

# 1. Diagnostic Search: Identify stack-allocated arrays or fixed-size buffers
echo "[*] Step 1: Scanning for raw stack arrays and fixed-size buffers..."
echo "------------------------------------------------------------------"
grep -rnE "\b(double|float|int|char|size_t)\s+\w+\s*\[\s*[0-9]+.*\]" src/ include/ tests/ || echo "No fixed-size stack arrays found with literal dimensions."
grep -rnE "alloca\(" src/ include/ tests/ || echo "No dynamic stack allocation (alloca) detected."

# 2. Smoking-Gun Source Audit: Inspect test file and core solver modules
echo ""
echo "[*] Step 2: Extracting source lines from test and solver modules..."
echo "------------------------------------------------------------------"
for file in tests/test_mass_continuity.cpp src/orchestrator.cpp src/pressure_poisson_solver.cpp; do
    if [ -f "$file" ]; then
        echo "=== FILE: $file ==="
        cat -n "$file" | head -n 100
    fi
done

# 3. Automated Repair Templates (Commented out for safety)
echo ""
echo "[*] Step 3: Available Automated Remediation Patches (Sed Injections)"
echo "------------------------------------------------------------------"
echo "If fixed-size stack arrays are identified, apply corrections using sed:"
echo "# sed -i 's/double \([a-zA-Z0-9_]*\)\[[0-9]*\]/std::vector<double> \\1(total_cells, 0.0)/g' src/orchestrator.cpp"
echo "# sed -i 's/int \([a-zA-Z0-9_]*\)\[[0-9]*\]/std::vector<int> \\1(total_cells, 0)/g' src/orchestrator.cpp"

echo "=================================================================="
echo "                  AUDIT COMPLETE - REVIEW OUTPUT                  "
echo "=================================================================="