#!/bin/bash
set -euo pipefail

echo "🌐 Installing C++ build essentials & testing libraries..."
sudo apt-get update
sudo apt-get install -y build-essential libgtest-dev

echo "🚀 Upgrading core Python packaging tools & installing pybind11..."
python -m pip install --upgrade pip setuptools wheel pybind11

echo "📦 Installing solver dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "⚙️ Compiling C++ solver extension via direct g++ with pybind11..."
mkdir -p bin
# Compile C++ kernel sources and operator implementations into a shared pybind11 module
g++ -O3 -march=native -shared -fPIC \
  cpp/src/*.cpp \
  cpp/src/ops/*.cpp \
  -Icpp/include \
  $(python3 -m pybind11 --includes) \
  -o bin/navier_stokes_solver$(python3-config --extension-suffix)

echo "✅ C++ native module compiled and ready for orchestration."