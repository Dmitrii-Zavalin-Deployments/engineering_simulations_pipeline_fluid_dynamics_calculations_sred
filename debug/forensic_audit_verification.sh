#!/bin/bash
# Description: Automated forensic audit for Navier-Stokes solver NaN propagation exception failure.
# Root Cause: Injection of NaN via boundary conditions bypassed exception gates because 
#             advection/stencil routines lacked explicit finite checks (std::isfinite).

echo "============================================================"
echo "🔍 STARTING DEEP FORENSIC AUDIT: Non-Finite / NaN Exception Failure"
echo "============================================================"

echo ""
echo "--- [DIAGNOSTIC 1] Searching for existing finite/NaN safety checks in C++ sources ---"
grep -rn "isfinite" cpp/src/ || echo "⚠️ Warning: No 'isfinite' checks located in cpp/src/."
grep -rn "isnan" cpp/src/ || echo "⚠️ Warning: No 'isnan' checks located in cpp/src/."

echo ""
echo "--- [DIAGNOSTIC 2] Inspecting Advection & Python Gate Implementation ---"
if [ -f "cpp/src/advection.cpp" ]; then
    echo "📄 Smoking-Gun Audit: cpp/src/advection.cpp (First 80 lines)"
    cat -n cpp/src/advection.cpp | head -n 80
else
    echo "❌ Critical: cpp/src/advection.cpp not found."
fi

if [ -f "cpp/src/python_gate.cpp" ]; then
    echo ""
    echo "📄 Smoking-Gun Audit: cpp/src/python_gate.cpp (First 80 lines)"
    cat -n cpp/src/python_gate.cpp | head -n 80
fi

echo ""
echo "============================================================"
echo "🛠️ RECOMMENDED REPAIR STRATEGY"
echo "============================================================"
echo "The Python tests expect a RuntimeError ('Advection term exploded in grid computation.')"
echo "when non-finite values are processed. Add a validation guard in advection.cpp or "
echo "python_gate.cpp to throw std::runtime_error when non-finite values are detected."
echo ""
echo "Example automated repair pattern (uncomment `# sed` below once targeting exact line):"

# # Example sed injection to insert a finite check inside advection.cpp loop or entry:
# sed -i '/v_z\[idx\];/a \    if (!std::isfinite(v_x[idx])) { throw std::runtime_error("Advection term exploded in grid computation."); }' cpp/src/advection.cpp

exit 1