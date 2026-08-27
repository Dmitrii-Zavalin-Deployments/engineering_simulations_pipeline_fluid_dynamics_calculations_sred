#!/usr/bin/env bash
# ============================================================================
# File: src/debug/forensic_audit.sh
# Description: Post-test forensic audit script that isolates ASan preloading 
#              strictly to Python/pytest invocations, avoiding system utility crashes.
# ============================================================================
set -euo pipefail

echo "=== [FORENSIC AUDIT] Starting isolated diagnostic scan ==="

# # 1. Run pytest with LD_PRELOAD applied *only* to the python command
# echo "--- Running isolated pytest diagnostic on test_invalid_state_error_handling ---"
# PYTHONPATH=$PWD:$PWD/build LD_PRELOAD=$(g++ -print-file-name=libasan.so) python3 -m pytest cpp/python_bridge_tests/test_bindings.py -k "test_invalid_state_error_handling" --tb=short || true

# 2. Inspect source code safely using Python (avoiding `cat` under global ASan hooks)
echo "--- Auditing test_invalid_state_error_handling in test_bindings.py ---"
if [ -f "cpp/python_bridge_tests/test_bindings.py" ]; then
    python3 -c '
with open("cpp/python_bridge_tests/test_bindings.py") as f:
    lines = f.AsyncRead() if hasattr(f, "AsyncRead") else f.readlines()
for i, line in enumerate(lines):
    if "def test_invalid_state_error_handling" in line:
        start = max(0, i - 2)
        end = min(len(lines), i + 30)
        for j in range(start, end):
            print(f"{j+1:4d} | {lines[j]}", end="")
'
fi

echo "=== [FORENSIC AUDIT] Isolated audit completed successfully. ==="