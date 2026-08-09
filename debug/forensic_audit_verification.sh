#!/bin/bash
set -e

echo "========================================================================="
echo "=== FORENSIC AUDIT: compute_trial_velocities Signature Mismatch ==="
echo "========================================================================="

echo "[1] Searching for compute_trial_velocities declarations and definitions..."
grep -rn "compute_trial_velocities" cpp/ || true

echo ""
echo "[2] Smoking-Gun Source Audit: cpp/include/predictor.hpp"
if [ -f cpp/include/predictor.hpp ]; then
    cat -n cpp/include/predictor.hpp | sed -n '20,38p'
else
    echo "WARNING: predictor.hpp not found."
fi

echo ""
echo "[3] Smoking-Gun Source Audit: cpp/src/orchestrator.cpp"
if [ -f cpp/src/orchestrator.cpp ]; then
    cat -n cpp/src/orchestrator.cpp | sed -n '25,45p'
else
    echo "WARNING: orchestrator.cpp not found."
fi

echo ""
echo "========================================================================="
echo "=== AUTOMATED REPAIR TEMPLATES (COMMENTED INJECTIONS) ==="
echo "========================================================================="
# Option A: Re-order predictor.hpp so double dt follows fluid consistently
# sed -i 's/const FluidProperties& fluid,/const FluidProperties& fluid, double dt,/g' cpp/include/predictor.hpp
# sed -i '/const std::vector<int>& mask,/ { N; /double dt,/d }' cpp/include/predictor.hpp

echo "=== FORENSIC AUDIT COMPLETE ==="