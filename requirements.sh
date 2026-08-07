#!/bin/bash
set -euo pipefail

echo "🌐 Installing C++ build essentials, testing libraries, and gcovr..."
sudo apt-get update
sudo apt-get install -y build-essential libgtest-dev gcovr

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

echo "🧪 Discovering and compiling native C++ unit tests dynamically..."
for test_file in tests/test_*.cpp; do
    if [ -f "$test_file" ]; then
        test_name=$(basename "$test_file" .cpp)
        echo "--- Compiling and executing: $test_name ---"
        g++ -O3 -march=native -std=c++17 \
          "$test_file" \
          cpp/src/*.cpp \
          cpp/src/ops/*.cpp \
          -Icpp/include \
          -lgtest -lgtest_main -pthread \
          -fopenmp \
          -o "bin/$test_name"
        
        # Execute the compiled test binary
        ./bin/$test_name
    fi
done

echo "✅ All C++ native modules compiled and unit tests passed successfully."