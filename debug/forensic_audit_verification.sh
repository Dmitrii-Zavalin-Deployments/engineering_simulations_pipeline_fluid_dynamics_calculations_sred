#!/usr/bin/env bash
set -euo pipefail

echo "=== [CLEANUP AUDIT] Searching for references to cpp/cpp_integration_tests/data ==="
grep -rn "cpp/cpp_integration_tests/data" . || echo "No exact string matches found."

echo "=== Searching for all file(COPY commands in CMake files ==="
find . -name "CMakeLists.txt" -exec grep -Hn "file(COPY" {} +

echo "=== Searching for any lingering 'data' directory references in tests or build scripts ==="
grep -rn "cpp_integration_tests" . || echo "No other cpp_integration_tests references found."