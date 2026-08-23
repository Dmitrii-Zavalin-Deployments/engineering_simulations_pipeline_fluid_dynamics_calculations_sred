#!/usr/bin/env bash
# src/debug/forensic_audit.sh
# Ultra‑forensic post‑test audit for MassContinuityTest stack‑smashing

set -euo pipefail

echo "[forensic] starting forensic_audit.sh"

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BUILD="${ROOT}/build"
LOGS="${ROOT}/src/debug/logs"
mkdir -p "${LOGS}"

CTEST_LOG="${LOGS}/ctest_full.log"
GTEST_LOG="${LOGS}/gtest_mass_continuity.log"

echo "[forensic] root: ${ROOT}"
echo "[forensic] build: ${BUILD}"
echo "[forensic] logs: ${LOGS}"

###############################################
# 1) Capture CTest output
###############################################
if [ -f "${BUILD}/Testing/Temporary/LastTest.log" ]; then
    cp "${BUILD}/Testing/Temporary/LastTest.log" "${CTEST_LOG}"
    echo "[forensic] saved LastTest.log -> ${CTEST_LOG}"
else
    echo "[forensic] no LastTest.log found" | tee "${CTEST_LOG}"
fi

###############################################
# 2) Extract smoking‑gun lines
###############################################
echo "[forensic] extracting MassContinuityTest diagnostics"
grep -n -E \
    "MassContinuityTest|MassContinuityIntegrationTest|stack smashing detected|Subprocess aborted|core dumped" \
    "${CTEST_LOG}" || true

###############################################
# 3) Re-run test with gtest flags
###############################################
BIN="${BUILD}/test_mass_continuity"

if [ -x "${BIN}" ]; then
    echo "[forensic] re-running ${BIN} with gtest flags"
    set +e
    "${BIN}" \
        --gtest_filter=MassContinuityIntegrationTest.EnforcesZeroDivergenceInFluidDomain \
        --gtest_repeat=1 \
        --gtest_break_on_failure \
        --gtest_print_time \
        2>&1 | tee "${GTEST_LOG}"
    EXIT=$?
    set -e
    echo "[forensic] MassContinuityIntegrationTest exit code: ${EXIT}"
else
    echo "[forensic] binary not found: ${BIN}"
fi

###############################################
# 4) Search for stack-smashing patterns
###############################################
echo "[forensic] scanning source for dangerous patterns"

for DIR in "${ROOT}/src" "${ROOT}/cpp" "${ROOT}/tests"; do
    if [ -d "${DIR}" ]; then
        echo "[forensic] scanning ${DIR}"
        grep -R --line-number --color=never -E \
            "strcpy|strncpy|sprintf|snprintf|memcpy|memmove|alloca|gets\(|std::array<|std::vector<|new " \
            "${DIR}" || true
    fi
done

###############################################
# 5) Identify MassContinuity-related files
###############################################
echo "[forensic] locating MassContinuity-related source files"
FILES=$(grep -R -l -E "MassContinuity|EnforcesZeroDivergenceInFluidDomain" "${ROOT}" || true)

###############################################
# 6) Dump numbered source for suspect files
###############################################
echo "[forensic] dumping numbered source for suspect files"
for F in ${FILES}; do
    if [ -f "${F}" ]; then
        echo "==================== [cat -n ${F}] ===================="
        cat -n "${F}"
    fi
done

###############################################
# 7) Extract binary symbols (nm + c++filt)
###############################################
if [ -x "${BIN}" ]; then
    echo "[forensic] extracting symbols from ${BIN}"
    nm -C "${BIN}" > "${LOGS}/nm_symbols.txt" || true
    echo "[forensic] nm symbols saved -> ${LOGS}/nm_symbols.txt"

    echo "[forensic] scanning for MassContinuity symbols"
    grep -n -E "MassContinuity|Divergence|BoundaryCondition" "${LOGS}/nm_symbols.txt" || true
fi

###############################################
# 8) Core dump analysis (if any)
###############################################
CORES=$(find "${BUILD}" -maxdepth 2 -type f -name "core*" 2>/dev/null || true)

if [ -n "${CORES}" ]; then
    echo "[forensic] core dump(s) detected:"
    echo "${CORES}"

    if command -v gdb >/dev/null 2>&1; then
        for C in ${CORES}; do
            echo "==================== [gdb backtrace for ${C}] ===================="
            gdb -q "${BIN}" "${C}" \
                -ex "set pagination off" \
                -ex "bt" \
                -ex "info registers" \
                -ex "info frame" \
                -ex "quit" || true
        done
    else
        echo "[forensic] gdb not available"
    fi
else
    echo "[forensic] no core dumps found"
fi

###############################################
# 9) Address-to-line mapping (addr2line)
###############################################
if command -v addr2line >/dev/null 2>&1 && [ -x "${BIN}" ]; then
    echo "[forensic] addr2line available — add addresses manually if logs show them"
    # Example:
    # ADDR="0x7ffdeadbeef"
    # echo "[forensic] mapping ${ADDR}"
    # addr2line -e "${BIN}" "${ADDR}"
fi

###############################################
# 10) ASAN-style red-zone check (best-effort)
###############################################
echo "[forensic] scanning logs for red-zone / heap-buffer-overflow hints"
grep -n -E "buffer overflow|heap-buffer-overflow|redzone|poisoned" "${CTEST_LOG}" || true
grep -n -E "buffer overflow|heap-buffer-overflow|redzone|poisoned" "${GTEST_LOG}" || true

###############################################
# 11) Suggested automated sed repairs (commented)
###############################################
echo "[forensic] providing sed auto-repair templates (commented)"

# sed -i 's/char buffer

\[128\]

/char buffer[256]  \/\/ expanded to avoid overflow/' "${ROOT}/cpp/mass_continuity_solver.cpp"
# sed -i 's/strcpy(/strncpy( \/\/ safer bounded copy/' "${ROOT}/cpp/mass_continuity_solver.cpp"
# sed -i 's/sprintf(/snprintf( \/\/ bounded formatting/' "${ROOT}/cpp/mass_continuity_solver.cpp"
# sed -i 's/std::array<double, 128>/std::array<double, 256> \/\/ increased stencil capacity/' "${ROOT}/cpp/mass_continuity_solver.hpp"
# sed -i 's/memcpy(/memmove( \/\/ safer for overlapping regions/' "${ROOT}/cpp/mass_continuity_solver.cpp"

###############################################
# 12) Final summary
###############################################
echo "[forensic] forensic_audit.sh completed"
echo "[forensic] artifacts:"
echo "  - ${CTEST_LOG}"
echo "  - ${GTEST_LOG}"
echo "  - ${LOGS}/nm_symbols.txt"
