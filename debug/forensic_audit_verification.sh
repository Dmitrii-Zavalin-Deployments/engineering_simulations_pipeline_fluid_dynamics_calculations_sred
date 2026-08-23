#!/usr/bin/env bash
# src/debug/forensic_audit.sh
# Post-test forensic audit for MassContinuityTest stack smashing in GitHub Actions

set -euo pipefail

echo "[forensic] starting forensic_audit.sh"

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BUILD_DIR="${ROOT_DIR}/build"
LOG_DIR="${ROOT_DIR}/src/debug/logs"
mkdir -p "${LOG_DIR}"

CTEST_LOG="${LOG_DIR}/ctest_full.log"
GTEST_LOG="${LOG_DIR}/gtest_mass_continuity.log"

echo "[forensic] root: ${ROOT_DIR}"
echo "[forensic] build: ${BUILD_DIR}"
echo "[forensic] logs: ${LOG_DIR}"

# 1) Capture raw CTest output (if available)
echo "[forensic] capturing CTest output (if any)"
if [ -f "${BUILD_DIR}/Testing/Temporary/LastTest.log" ]; then
  cp "${BUILD_DIR}/Testing/Temporary/LastTest.log" "${CTEST_LOG}"
  echo "[forensic] saved LastTest.log -> ${CTEST_LOG}"
else
  echo "[forensic] no LastTest.log found in ${BUILD_DIR}/Testing/Temporary" | tee "${CTEST_LOG}"
fi

# 2) Extract MassContinuityTest section + stack smashing lines
echo "[forensic] extracting MassContinuityTest diagnostics"
grep -n -E "MassContinuityTest|MassContinuityIntegrationTest|stack smashing detected|Subprocess aborted" "${CTEST_LOG}" || true

# 3) Run MassContinuityTest directly with verbose output (if binary exists)
MASS_BIN="${BUILD_DIR}/test_mass_continuity"
if [ -x "${MASS_BIN}" ]; then
  echo "[forensic] re-running ${MASS_BIN} with --gtest_filter and --gtest_break_on_failure"
  set +e
  "${MASS_BIN}" \
    --gtest_filter=MassContinuityIntegrationTest.EnforcesZeroDivergenceInFluidDomain \
    --gtest_repeat=1 \
    --gtest_break_on_failure \
    --gtest_print_time \
    2>&1 | tee "${GTEST_LOG}"
  MASS_EXIT=$?
  set -e
  echo "[forensic] MassContinuityIntegrationTest exit code: ${MASS_EXIT}"
else
  echo "[forensic] binary not found: ${MASS_BIN}"
fi

# 4) Search for typical stack-smashing root causes in source tree
echo "[forensic] scanning source for potential stack-smashing patterns"
SRC_DIR="${ROOT_DIR}/src"
TEST_DIR="${ROOT_DIR}/tests"

for DIR in "${SRC_DIR}" "${TEST_DIR}"; do
  if [ -d "${DIR}" ]; then
    echo "[forensic] scanning ${DIR}"
    grep -R --line-number --color=never -E \
      "strcpy|strncpy|memcpy|memmove|sprintf|snprintf|gets\(|fgets\(|std::array<|std::vector<|new " \
      "${DIR}" || true
  else
    echo "[forensic] directory not found: ${DIR}"
  fi
done

# 5) Identify likely smoking-gun files around MassContinuity* symbols
echo "[forensic] locating MassContinuity-related source files"
grep -R --line-number --color=never -E "MassContinuity|EnforcesZeroDivergenceInFluidDomain" "${ROOT_DIR}" || true

# 6) cat -n suspected source files for manual audit
echo "[forensic] dumping numbered source for MassContinuity-related files"
SUSPECT_FILES=$(grep -R -l -E "MassContinuity|EnforcesZeroDivergenceInFluidDomain" "${ROOT_DIR}" || true)

if [ -n "${SUSPECT_FILES}" ]; then
  for F in ${SUSPECT_FILES}; do
    if [ -f "${F}" ]; then
      echo "==================== [cat -n ${F}] ===================="
      cat -n "${F}"
    fi
  done
else
  echo "[forensic] no MassContinuity-related files found via grep"
fi

# 7) If core dump exists, try to map crash location (best-effort)
CORE_CANDIDATES=$(find "${BUILD_DIR}" -maxdepth 2 -type f -name "core*" 2>/dev/null || true)
if [ -n "${CORE_CANDIDATES}" ]; then
  echo "[forensic] core dump(s) detected:"
  echo "${CORE_CANDIDATES}"
  # NOTE: GitHub Actions may not have gdb installed; this is best-effort.
  if command -v gdb >/dev/null 2>&1; then
    for CORE in ${CORE_CANDIDATES}; do
      echo "==================== [gdb backtrace for ${CORE}] ===================="
      gdb -q "${MASS_BIN}" "${CORE}" -ex "bt" -ex "info frame" -ex "quit" || true
    done
  else
    echo "[forensic] gdb not available; skipping core backtrace"
  fi
else
  echo "[forensic] no core dumps found under ${BUILD_DIR}"
fi

# 8) addr2line mapping for last frames (if we have an address hint)
# (Placeholder: if you capture an address from logs, you can feed it here.)
if command -v addr2line >/dev/null 2>&1 && [ -x "${MASS_BIN}" ]; then
  echo "[forensic] addr2line is available; add addresses here if logs expose them"
  # Example usage (manual injection once you know the address):
  # echo "[forensic] mapping example address 0xDEADBEEF"
  # addr2line -e "${MASS_BIN}" 0xDEADBEEF || true
fi

# 9) Suggested automated repair sed injections (commented out)
# These are templates; enable and adapt once the smoking-gun lines are known.

# sed -i 's/char buffer

\[128\]

;/char buffer[256]; \/\/ expanded to avoid overflow/' "${SRC_DIR}/mass_continuity_solver.cpp"
# sed -i 's/strcpy(\(.*\));/strncpy(\1, sizeof(\1)); \/\/ safer copy with bound/' "${SRC_DIR}/mass_continuity_solver.cpp"
# sed -i 's/sprintf(\(.*\));/snprintf(\1, sizeof(\1)); \/\/ bounded sprintf/' "${SRC_DIR}/mass_continuity_solver.cpp"
# sed -i 's/std::array<double, 128>/std::array<double, 256> \/\/ increased capacity for divergence stencil/' "${SRC_DIR}/mass_continuity_solver.hpp"
# sed -i 's/memcpy(\(.*\));/memmove(\1); \/\/ safer move for overlapping regions/' "${SRC_DIR}/mass_continuity_solver.cpp"

# 10) Final summary
echo "[forensic] forensic_audit.sh completed"
echo "[forensic] artifacts:"
echo "  - CTest log:  ${CTEST_LOG}"
echo "  - GTest log:  ${GTEST_LOG}"
