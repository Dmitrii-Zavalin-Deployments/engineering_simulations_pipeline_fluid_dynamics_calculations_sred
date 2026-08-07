#!/bin/bash
set -euo pipefail

echo "========================================================="
echo "🔍 FORENSIC AUDIT: Pybind11 Inclusion & Compilation Failure"
echo "========================================================="

echo "--- 1. Diagnostic: Check Python environment and pybind11 module ---"
python3 -c "import sys; print('Python executable:', sys.executable)"
python3 -c "import pybind11; print('pybind11 include dir:', pybind11.get_include())" || echo "⚠️ pybind11 Python package not detected."

echo "--- 2. Diagnostic: Scan system for pybind11 headers ---"
find /usr -name "pybind11.h" 2>/dev/null || echo "⚠️ pybind11.h not found under /usr."

echo "--- 3. Smoking-Gun Source Audit: requirements.sh ---"
if [ -f "requirements.sh" ]; then
    cat -n requirements.sh
else
    echo "❌ requirements.sh not found in workspace root."
fi

echo "--- 4. Automated Repair Prescription (Commented Sed Injections) ---"
# To incorporate pybind11 include paths into the direct g++ compilation command, apply the following patch:
# sed -i 's|g++ -O3 -march=native cpp/src/\*.cpp -Icpp/include -o bin/navier_stokes_solver|g++ -O3 -march=native cpp/src/*.cpp -Icpp/include $(python3 -m pybind11 --includes) -o bin/navier_stokes_solver|g' requirements.sh

# To ensure the build container installs python3-pybind11 alongside build essentials, apply:
# sed -i '/sudo apt-get install -y build-essential libgtest-dev/s/libgtest-dev/libgtest-dev python3-pybind11/' requirements.sh

echo "========================================================="
echo "🏁 Forensic Audit Complete."
echo "========================================================="