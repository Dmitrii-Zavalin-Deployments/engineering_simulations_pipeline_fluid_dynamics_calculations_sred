#!/usr/bin/env bash
# ==============================================================================
# @file parse_test_failure.sh
# @brief Automated CFD Test Log Parser & Forensic Diagnostic Tool
# @details Executes CTest, captures raw verbose output, isolates failed test cases,
#          extracts Google Test assertion failures, and generates a structured summary.
# ==============================================================================

set -eo pipefail

BUILD_DIR="${1:-build}"
LOG_FILE="test_execution.log"
REPORT_FILE="test_failure_report.txt"

if [ ! -d "${BUILD_DIR}" ]; then
    echo "[ERROR] Build directory '${BUILD_DIR}' does not exist. Run cmake/make first."
    exit 1
fi

echo "[INFO] Running test suite from '${BUILD_DIR}'..."
cd "${BUILD_DIR}"

# Run CTest with maximum verbosity and capture output
ctest --output-on-failure --verbose 2>&1 | tee "../${LOG_FILE}" || true
cd ..

echo "[INFO] Parsing execution log for root cause analysis..."

python3 - << 'EOF' > "${REPORT_FILE}"
import re

try:
    with open("test_execution.log", "r") as f:
        log_content = f.read()
except FileNotFoundError:
    print("[FATAL] Test execution log not found.")
    exit(1)

print("========================================================")
print("          CFD TEST SUITE FORENSIC AUDIT REPORT          ")
print("========================================================")

# Extract summary of failed tests
failed_matches = re.findall(r'\[\s+FAILED\s+\]\s+([^\s]+)', log_content)
unique_failures = list(dict.fromkeys(failed_matches))

print(f"\n[SUMMARY] Total Test Failures: {len(unique_failures)}")
for test in unique_failures:
    print(f"  * Failed Test: {test}")

print("\n--------------------------------------------------------")
print("[DIAGNOSTICS] Extracted Assertion Failures & Stack Traces:")
print("--------------------------------------------------------")

# Split log by individual test runs to isolate failures
test_blocks = log_content.split("[ RUN      ]")
for block in test_blocks:
    if any(keyword in block for keyword in ["[  FAILED  ]", "[FATAL]", "Assertion", "exceeded", "Error"]):
        lines = block.strip().split("\n")
        test_header = lines[0] if lines else "Unknown Test"
        print(f"\n[TestCase] {test_header}")
        
        for line in lines:
            trimmed = line.strip()
            # Filter for relevant diagnostic markers
            if any(marker in trimmed for marker in [
                "[  FAILED  ]", "[FATAL]", "Expected", "Actual", 
                "Error", "Divergence", "Residue", "Relative L2", 
                "velocity", "overflow", "NaN", "inf"
            ]):
                print(f"    -> {trimmed}")

print("\n--------------------------------------------------------")
print("[ENVIRONMENT] Threading & Runtime Configuration:")
print("--------------------------------------------------------")
for line in log_content.split("\n"):
    if any(env_key in line for env_key in ["THREADING", "OpenMP", "Hardware", "OMP_"]):
        print(f"    {line.strip()}")

EOF

echo "[INFO] Forensic report successfully generated: ${REPORT_FILE}"
echo "--------------------------------------------------------"
cat "${REPORT_FILE}"