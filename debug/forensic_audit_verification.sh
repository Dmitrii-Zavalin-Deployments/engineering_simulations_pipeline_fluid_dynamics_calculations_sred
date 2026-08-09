#!/bin/bash
set -e

echo "========================================================================="
echo "=== FORENSIC AUDIT: FluidProperties Struct & Predictor Interface ==="
echo "========================================================================="

echo "[1] Locating FluidProperties definition across repository headers..."
grep -rn "struct FluidProperties" cpp/ || true

echo ""
echo "[2] Smoking-Gun Source Audit: Header file defining FluidProperties"
HEADER_FILE=$(grep -rl "struct FluidProperties" cpp/ | head -n 1)
if [ -n "$HEADER_FILE" ]; then
    echo "Found definition in: $HEADER_FILE"
    cat -n "$HEADER_FILE" | grep -A 12 "struct FluidProperties"
else
    echo "WARNING: Could not locate FluidProperties definition."
fi

echo ""
echo "[3] Smoking-Gun Source Audit: cpp/src/predictor.cpp (Lines 110-125)"
if [ -f cpp/src/predictor.cpp ]; then
    cat -n cpp/src/predictor.cpp | sed -n '110,125p'
else
    echo "WARNING: cpp/src/predictor.cpp not found."
fi

echo ""
echo "[4] Smoking-Gun Source Audit: Orchestrator invocation of predictor"
if [ -f cpp/src/orchestrator.cpp ]; then
    grep -n -C 5 "compute_trial_velocities" cpp/src/orchestrator.cpp || true
fi

echo ""
echo "========================================================================="
echo "=== AUTOMATED REPAIR TEMPLATES (COMMENTED INJECTIONS) ==="
echo "========================================================================="
# Option A: Revert predictor.cpp force terms back to raw fx/fy/fz (if forces are acceleration vectors)
# sed -i 's/fx\[idx\] \/ fluid.density/fx\[idx\]/g' cpp/src/predictor.cpp
# sed -i 's/fy\[idx\] \/ fluid.density/fy\[idx\]/g' cpp/src/predictor.cpp
# sed -i 's/fz\[idx\] \/ fluid.density/fz\[idx\]/g' cpp/src/predictor.cpp

# Option B: Add density field directly to FluidProperties struct if needed by interface contract
# sed -i '/struct FluidProperties {/a \    double density = 1000.0;' cpp/src/predictor.hpp

echo "=== FORENSIC AUDIT COMPLETE ==="