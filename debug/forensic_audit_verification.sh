#!/usr/bin/env bash
# ==============================================================================
# Forensic Audit Script for Navier-Stokes Solver Build Failures
# Location: src/debug/forensic_audit.sh
# ==============================================================================

set -euo pipefail

echo "=================================================================="
echo "[1/3] RUNNING CODE & OUTPUT DIAGNOSTICS (GREP / CAT)"
echo "=================================================================="

echo "-> Inspecting advection header declarations:"
if [ -f "cpp/include/advection.hpp" ]; then
    cat -n cpp/include/advection.hpp
else
    echo "WARNING: cpp/include/advection.hpp not found."
fi

echo "-> Inspecting laplacian header declarations:"
if [ -f "cpp/include/laplacian.hpp" ]; then
    cat -n cpp/include/laplacian.hpp
else
    echo "WARNING: cpp/include/laplacian.hpp not found."
fi

echo "=================================================================="
echo "[2/3] SMOKING-GUN SOURCE AUDIT (PREDICTOR KERNEL)"
echo "=================================================================="

echo "-> Inspecting cpp/src/predictor.cpp lines 50 to 90 where signature mismatches occur:"
if [ -f "cpp/src/predictor.cpp" ]; then
    sed -n '50,90p' cpp/src/predictor.cpp
else
    echo "ERROR: cpp/src/predictor.cpp not found."
fi

echo "=================================================================="
echo "[3/3] AUTOMATED REPAIR INJECTIONS (SED TEMPLATES)"
echo "=================================================================="
echo "Root Cause Identified:"
echo "  1. Advection API mismatch: compute_advection_x/y/z do not match repository headers."
echo "  2. Laplacian API mismatch: compute_laplacian is a void function taking output buffers."
echo "Below are the corresponding automated fix templates (commented out per specification):"

# sed -i 's/double adv_x = compute_advection_x(.*/double adv_x = compute_advection(u, v, w, i, j, k, dims.dx, dims.dy, dims.dz, ny, nz);/g' cpp/src/predictor.cpp
# sed -i 's/double lap_u = compute_laplacian(.*/double lap_u = 0.0; compute_laplacian(u, &lap_u, i, j, k, dims.dx, dims.dy, dims.dz);/g' cpp/src/predictor.cpp

echo "=== Forensic Audit Complete ==="