#!/usr/bin/env bash
# ==============================================================================
# @file src/debug/forensic_audit.sh
# @brief Post-test forensic diagnostic and automated repair script for GitHub Actions
# ==============================================================================

set -euo pipefail


# Grep for nested with statement patterns in test files
echo "Searching for potential nested 'with' statements across test suite..."
grep -rn "with " tests/ --include="*.py" | head -n 20 || true

echo ""
echo "=============================================================================="
echo "SMOKING-GUN SOURCE AUDIT: Line-Numbered Inspection of tests/test_cpp_gate.py"
echo "=============================================================================="

# Display lines 235 through 248 of tests/test_cpp_gate.py where SIM117 was flagged
cat -n tests/test_cpp_gate.py | sed -n '235,248p'

echo ""
echo "=============================================================================="
echo "AUTOMATED REPAIR INJECTIONS (SIM117 Compliance)"
echo "=============================================================================="

# Apply automated sed repair to combine nested with statements into a single multi-context with statement
sed -i '240,242s/.*/    with patch.dict(sys.modules, {"navier_stokes_cpp": None}), pytest.raises(ImportError, match=r"Failed to import compiled C\\+\\+ module \\'navier_stokes_cpp\\'"):/' tests/test_cpp_gate.py

echo "Verifying repair result:"
cat -n tests/test_cpp_gate.py | sed -n '238,246p'

echo ""
echo "=============================================================================="
echo "Forensic audit and diagnostic sequence completed."