#!/bin/bash
set -e

echo "=== 2. Inspecting python_gate.cpp for Field Synchronization and Step Invocation ==="
if [ -f "./cpp/src/python_gate.cpp" ]; then
    cat -n ./cpp/src/python_gate.cpp || true
fi

echo "=== 3. Inspecting orchestrator.cpp for Solver Step Orchestration ==="
if [ -f "./cpp/src/orchestrator.cpp" ]; then
    cat -n ./cpp/src/orchestrator.cpp || true
fi