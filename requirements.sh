#!/bin/bash
set -euo pipefail

echo "🌐 Installing C++ build essentials, CMake, testing libraries, and gcovr..."
sudo apt-get update
sudo apt-get install -y build-essential cmake libgtest-dev nlohmann-json3-dev gcovr

echo "🚀 Upgrading core Python packaging tools & installing pybind11..."
python -m pip install --upgrade pip setuptools wheel pybind11

echo "📦 Installing solver dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "⚙️ Configuring and building all targets via CMake..."
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel $(nproc)

echo "✅ Environment setup and CMake compilation completed successfully."
