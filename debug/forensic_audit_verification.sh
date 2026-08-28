#!/usr/bin/env bash
# ==============================================================================
# File: debug/find_test_failure.sh
# Description: Locate the exact test file and line 351 failure context.
# ==============================================================================

set -euo pipefail

echo "=== [STEP 1] Finding test files ==="
find cpp/tests -type f

echo "=== [STEP 2] Inspecting test files around line 351 ==="
TEST_FILE=$(find cpp/tests -name "*.cpp" | xargs grep -n "351" || true)
if [ -n "$TEST_FILE" ]; then
    echo "Found reference to line 351:"
    echo "$TEST_FILE"
else
    echo "Searching for test files containing assertions..."
    for f in $(find cpp/tests -name "*.cpp"); do
        if [ $(wc -l < "$f") -ge 350 ]; then
            echo "File with >= 350 lines: $f"
            cat -n "$f" | sed -n '335,365p'
        fi
    done
fi

echo "=== [STEP 3] Inspecting pressure poisson solver implementation ==="
POISSON_FILE=$(find cpp/src -name "*poisson*.cpp" 2>/dev/null | head -n 1)
if [ -n "$POISSON_FILE" ]; then
    echo "Found: $POISSON_FILE"
    cat -n "$POISSON_FILE" | head -n 120
else
    echo "Poisson file not found."
fi