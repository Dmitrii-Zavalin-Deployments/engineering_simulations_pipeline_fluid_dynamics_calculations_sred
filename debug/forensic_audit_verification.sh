#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
TEST_BINARY="test_full_pipeline_constant_flow"
TARGET_I=2
TARGET_J=2
TARGET_K=1
NX=8
NY=8

# Calculate flat index based on repository standard: i + nx * (j + ny * k)
FLAT_IDX=$(( TARGET_I + NX * (TARGET_J + NY * TARGET_K) ))

echo "=== NAVIER-STOKES PIPELINE FORENSIC DIAGNOSTIC ==="
echo "Target Cell: (${TARGET_I}, ${TARGET_J}, ${TARGET_K}) -> Flat Index: ${FLAT_IDX}"

if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake -DCMAKE_BUILD_TYPE=Debug ..
else
    cd "$BUILD_DIR"
fi

echo "Building test target..."
cmake --build . --target "$TEST_BINARY" -j$(nproc)

EXECUTABLE="./${TEST_BINARY}"
if [ ! -f "$EXECUTABLE" ]; then
    EXECUTABLE=$(find . -name "$TEST_BINARY" -type f -executable | head -n 1)
fi

echo "Executing test and capturing debug snapshots..."
OUTPUT_LOG="pipeline_debug_run.log"
"$EXECUTABLE" --gtest_filter=*StepByStepMicroManaged* > "$OUTPUT_LOG" 2>&1 || true

echo "=== EXTRACTING SNAPSHOT METRICS FOR CELL ${FLAT_IDX} ==="
python3 -c "
import os

flat_idx = ${FLAT_IDX}
log_path = '${OUTPUT_LOG}'

if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    print(f'Total log lines captured: {len(lines)}')
    print('--- Relevant Failure & Snapshot Events ---')
    for line in lines:
        if any(kw in line for kw in ['Failure', 'exceeds', 'DEBUG_DUMP', 'u[idx]', 'Non-zero']):
            print(line.strip())
else:
    print('Error: Log file not found.')
"

echo "Diagnostic capture complete. Inspect full trace in ${BUILD_DIR}/${OUTPUT_LOG}"