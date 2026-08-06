#!/bin/bash
# Description: Automated forensic audit for CMake build failures and missing CMakeLists.txt
# Usage: bash debug/forensic_audit.sh

echo "============================================================"
echo "🔍 STARTING CMAKE BUILD FORENSIC AUDIT"
echo "============================================================"

echo "📂 1. Current Working Directory & File Listing:"
pwd
ls -la

echo "============================================================"
echo "🔎 2. Searching for CMakeLists.txt across the repository:"
find . -name "CMakeLists.txt" -maxdepth 3 || echo "❌ No CMakeLists.txt found!"

echo "============================================================"
echo "📜 3. Auditing build execution scripts (requirements.sh / build scripts):"
if [ -f "requirements.sh" ]; then
  echo "--- requirements.sh contents (with line numbers) ---"
  cat -n requirements.sh
else
  echo "⚠️ requirements.sh not found in root."
fi

echo "============================================================"
echo "🔎 4. Scanning for CMake command invocations:"
grep -rn "cmake" . --exclude-dir=.git --exclude=debug/ || echo "No explicit cmake calls found in tracked files."

echo "============================================================"
echo "🛠️ 5. Automated Repair Suggestions (Commented Out):"
# If CMakeLists.txt is in a subdirectory (e.g., src/ or cpp/), adjust the build invocation:
# sed -i 's|cmake \.\.|cmake ./src|g' requirements.sh
# If CMakeLists.txt is missing entirely from the root, create a symlink or copy it:
# ln -s src/CMakeLists.txt ./CMakeLists.txt

echo "============================================================"
echo "🏁 Forensic Audit Complete."
exit 0