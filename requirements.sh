#!/bin/bash
set -euo pipefail

echo "🌐 Installing C++ build dependencies..."
sudo apt-get update
sudo apt-get install -y cmake build-essential libgtest-dev

echo "🚀 Upgrading core Python packaging tools..."
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installing solver dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "⚙️ Building C++ native kernels & extensions via CMake..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..