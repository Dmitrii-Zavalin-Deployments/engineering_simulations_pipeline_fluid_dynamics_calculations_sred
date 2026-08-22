#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo " [FORENSIC AUDIT] Core Library Macro Propagation Diagnostic"
echo "=================================================="

ROOT_CMAKE="CMakeLists.txt"

# 1. Diagnostics: Search for navier_stokes_core target and compile definitions
echo "--> Step 1: Running grep diagnostics for navier_stokes_core and target definitions..."
grep -rn "navier_stokes_core" . || echo "navier_stokes_core target not found in grep."
grep -rn "add_library" . || echo "add_library not found."

# 2. Source Audit: Inspect root CMakeLists.txt or library definition
echo "--> Step 2: Executing cat -n source audit on root CMakeLists.txt..."
if [ -f "$ROOT_CMAKE" ]; then
    cat -n "$ROOT_CMAKE" | head -n 60
else
    echo "ERROR: Root CMakeLists.txt does not exist." >&2
    exit 1
fi

# 3. Automated Repair Injections
echo "--> Step 3: Providing safe automated repair injection templates..."
echo "    (Uncomment the sed command below in CI/CD pipeline or local runner to apply fix)"

# # Example repair: Add global compile definitions in root CMakeLists.txt so navier_stokes_core and tests compile with debug snapshot tracking enabled
# sed -i '/project(/a add_compile_definitions(NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS)' CMakeLists.txt

echo "=================================================="
echo " [FORENSIC AUDIT] Audit complete."
echo "=================================================="