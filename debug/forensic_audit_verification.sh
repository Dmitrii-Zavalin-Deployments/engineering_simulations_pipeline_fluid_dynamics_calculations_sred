#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
TARGET_I=2
TARGET_J=2
TARGET_K=1
NX=8
NY=8
FLAT_IDX=$(( TARGET_I + NX * (TARGET_J + NY * TARGET_K) ))

LOG_PATH="${BUILD_DIR}/pipeline_debug_run.log"

if [ ! -f "$LOG_PATH" ]; then
    echo "Error: ${LOG_PATH} not found. Run the test first."
    exit 1
}

echo "=== DETAILED TRACE FOR CELL ${FLAT_IDX} (i=${TARGET_I}, j=${TARGET_J}, k=${TARGET_K}) ==="
python3 -c "
import os

flat_idx = ${FLAT_IDX}
log_path = '${LOG_PATH}'

with open(log_path, 'r') as f:
    content = f.read()

print('Searching log for snapshot dumps and vector states...')
# Let's inspect all lines containing snapshot or array data around cell ${FLAT_IDX}
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'DEBUG_DUMP' in line or 'Snapshot' in line or 'u[' in line or 'p[' in line:
        print(line)
        # Print surrounding lines if any vector values are printed
        for j in range(max(0, i-2), min(len(lines), i+3)):
            if j != i and ('[' in lines[j] or 'val' in lines[j] or 'cell' in lines[j]):
                print(f'   {lines[j]}')
"