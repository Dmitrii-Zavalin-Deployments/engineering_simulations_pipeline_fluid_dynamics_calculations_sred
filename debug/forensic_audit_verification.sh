#!/usr/bin/env bash
set -euo pipefail

echo "=== INSPECTING DIVERGENCE AND POISSON SOLVER IMPLEMENTATIONS ==="

python3 -c "
import os

files_to_check = [
    'cpp/src/ops/divergence.cpp',
    'cpp/include/divergence.hpp',
    'cpp/src/pressure_poisson_solver.cpp',
    'cpp/include/pressure_poisson_solver.hpp'
]

for fpath in files_to_check:
    if os.path.exists(fpath):
        print(f'=== {fpath} ===')
        with open(fpath, 'r') as f:
            print(f.read())
    else:
        print(f'=== {fpath} NOT FOUND ===')
"