#!/usr/bin/env bash
set -euo pipefail

TEST_FILE="cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp"

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: ${TEST_FILE} not found."
    exit 1
fi

echo "Extracting lines around line 351 in ${TEST_FILE}..."

python3 -c "
file_path = '${TEST_FILE}'
with open(file_path, 'r') as f:
    lines = f.readlines()

start_idx = max(0, 335)
end_idx = min(len(lines), 365)

for idx in range(start_idx, end_idx):
    print(f'{idx+1}: {lines[idx].rstrip()}')
"