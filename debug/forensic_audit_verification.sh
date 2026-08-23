#!/usr/bin/env bash
# ==============================================================================
# Forensic Audit Script: Stack Smashing Analysis (MassContinuityTest)
# Target: cpp/src/pressure_poisson_solver.cpp
# ==============================================================================
set -euo pipefail

echo "=================================================================="
echo "🔍 STARTING FORENSIC AUDIT: Stack Smashing & Poisson Solver Analysis"
echo "=================================================================="

echo "--- 1. Diagnostic Environment & Compiler Flags ---"
echo "Checking build configuration and compiler flags for stack protection..."
if [ -f "build/CMakeCache.txt" ]; then
    grep -i "CMAKE_CXX_FLAGS" build/CMakeCache.txt || echo "No CMAKE_CXX_FLAGS in cache."
else
    echo "build/CMakeCache.txt not found. Inspecting CMakeLists.txt instead:"
    grep -rn "CXX_FLAGS" CMakeLists.txt cpp/ || echo "No explicit CXX_FLAGS found in CMake files."
fi

echo ""
echo "--- 2. Scanning Pressure Poisson Solver for Indexing & Memory Risks ---"
echo "Checking array access patterns, loops, and memory operations in pressure_poisson_solver.cpp:"
grep -n -E "(vector|\[|\]|memcpy|memset|get_flat_index)" cpp/src/pressure_poisson_solver.cpp || true

echo ""
echo "--- 3. Smoking-Gun Source Audit: pressure_poisson_solver.cpp ---"
POISSON_SRC="cpp/src/pressure_poisson_solver.cpp"
if [ -f "$POISSON_SRC" ]; then
    echo "Found source file: $POISSON_SRC. Displaying complete source with line numbers:"
    cat -n "$POISSON_SRC"
else
    echo "⚠️ Error: $POISSON_SRC not found."
    find cpp/src -name "*.cpp"
fi

echo ""
echo "=================================================================="
echo "🛠️ AUTOMATED REPAIR TEMPLATES (Uncomment `# sed` lines to apply fixes)"
echo "=================================================================="
# Fix potential boundary off-by-one or array bounds condition in solver loops:
# sed -i 's/nx - 1/nx - 2/g' cpp/src/pressure_poisson_solver.cpp
# sed -i 's/ny - 1/ny - 2/g' cpp/src/pressure_poisson_solver.cpp
# sed -i 's/nz - 1/nz - 2/g' cpp/src/pressure_poisson_solver.cpp

# Fix OpenMP conditional threshold for small grids (e.g., 1000 cells):
# sed -i 's/total_cells > 1000/total_cells >= 1000/g' cpp/src/pressure_poisson_solver.cpp

echo "Forensic audit complete."