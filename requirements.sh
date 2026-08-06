#!/bin/bash
set -euo pipefail

echo "🌐 Installing C++ build essentials & testing libraries..."
sudo apt-get update
sudo apt-get install -y build-essential libgtest-dev

echo "🚀 Upgrading core Python packaging tools..."
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installing solver dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "⚙️ Compiling C++ solver binary via direct g++..."
mkdir -p bin
# Compile C++ source files directly into a high-performance standalone binary
g++ -O3 -march=native src/*.cpp -o bin/navier_stokes_solver

echo "✅ C++ native binary compiled and ready for orchestration."