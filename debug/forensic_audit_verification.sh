#!/usr/bin/env bash
# ==============================================================================
# File: src/debug/forensic_audit.sh
# Description: Post-test forensic script for GitHub Actions CI failure analysis.
# ==============================================================================

set -uo pipefail

echo "================================================================="
echo "  GITHUB ACTIONS FORENSIC AUDIT: STACK SMASHING & OMP DIAGNOSTICS"
echo "================================================================="

echo -e "\n[+] System & Environment Telemetry:"
uname -a
echo "Core dump status:"
ulimit -c

echo -e "\n[+] OpenMP & Thread Stack Configuration:"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-Not Set}"
echo "OMP_STACKSIZE:   ${OMP_STACKSIZE:-Not Set}"
lscpu | grep -E "CPU\(s\):|Thread|Core|Socket"

echo -e "\n[+] Scanning Build Directory and Test Binaries:"
find build/ -maxdepth 2 -type f -executable

echo -e "\n[+] Inspecting Last CTest Log Output / XML Reports:"
if [ -d "build/Testing" ]; then
    find build/Testing -name "*.xml" -o -name "*.log" | while read -r logfile; do
        echo "--- Found log: $logfile ---"
        tail -n 50 "$logfile"
    done
else
    echo "    No CTest Testing directory found."
end

echo -e "\n[+] Checking System Kernel Messages for Segmentation Faults / Aborts:"
dmesg -T | tail -n 30 || echo "    dmesg restricted or unavailable."

echo -e "\n[+] Forensic audit diagnostic complete."