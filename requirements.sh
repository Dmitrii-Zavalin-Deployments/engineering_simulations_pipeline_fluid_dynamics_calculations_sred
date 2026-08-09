#!/bin/bash
set -euo pipefail

echo "🌐 Installing C++ build essentials, testing libraries, and gcovr..."
sudo apt-get update
sudo apt-get install -y build-essential libgtest-dev gcovr

echo "🚀 Upgrading core Python packaging tools & installing pybind11..."
python -m pip install --upgrade pip setuptools wheel pybind11

echo "📦 Installing solver dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "⚙ Compiling base C++ solver extension..."
mkdir -p bin
# Compile core sources and nested ops/ directory using find to ensure complete coverage
g++ -O3 -march=native -shared -fPIC \
  $(find cpp/src -name "*.cpp") \
  -Icpp/include \
  $(python3 -m pybind11 --includes) \
  -o navier_stokes_cpp$(python3-config --extension-suffix)

echo "✅ Base environment and C++ module compiled successfully."