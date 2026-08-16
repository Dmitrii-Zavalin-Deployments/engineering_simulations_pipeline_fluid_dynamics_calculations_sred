#!/usr/bin/env bash
# ==============================================================================
# @file src/debug/forensic_audit.sh
# @brief Post-test forensic diagnostic and automated repair script for GitHub Actions
# ==============================================================================

set -euo pipefail

echo "Searching for potential nested 'with' statements in tests/test_main.py..."
grep -rn "with " tests/test_main.py || true

echo ""
echo "=============================================================================="
echo "SMOKING-GUN SOURCE AUDIT: Line-Numbered Inspection of tests/test_main.py"
echo "=============================================================================="

# Inspect lines 165 through 185 and 200 through 220
echo "Inspecting lines 165 through 185:"
cat -n tests/test_main.py | sed -n '165,185p'

echo ""
echo "Inspecting lines 200 through 220:"
cat -n tests/test_main.py | sed -n '200,220p'

echo ""
echo "=============================================================================="
echo "AUTOMATED REPAIR INJECTIONS (SIM117 Compliance via Python Refactoring)"
echo "=============================================================================="

python3 -c '
path = "tests/test_main.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix occurrence around line 170 (Solver diverged)
target_170 = """    with patch("src.main.BASE_DIR", tmp_path), \\
         patch("src.main.load_and_validate_inputs", return_value=({}, {})), \\
         patch("src.main.SolverState", return_value=mock_state), \\
         patch("src.main.step_simulation", side_effect=RuntimeError("Solver diverged")):

         with pytest.raises(RuntimeError, match="Solver diverged"):
             run_simulation(
                 input_output_folder=tmp_path,
                 input_file_name=input_file.name,
                 output_file_name="output_manifest.json",
             )"""

replacement_170 = """    with patch("src.main.BASE_DIR", tmp_path), \\
         patch("src.main.load_and_validate_inputs", return_value=({}, {})), \\
         patch("src.main.SolverState", return_value=mock_state), \\
         patch("src.main.step_simulation", side_effect=RuntimeError("Solver diverged")), \\
         pytest.raises(RuntimeError, match="Solver diverged"):
         run_simulation(
             input_output_folder=tmp_path,
             input_file_name=input_file.name,
             output_file_name="output_manifest.json",
         )"""

# Fix occurrence around line 204 (Divergent field value)
target_204 = """    with patch("src.main.BASE_DIR", tmp_path), \\
         patch("src.main.load_and_validate_inputs", return_value=({}, {})), \\
         patch("src.main.SolverState", return_value=mock_state), \\
         patch("src.main.step_simulation", side_effect=ValueError("Divergent field value")), \\
         patch("src.main.archive_simulation_results", side_effect=OSError("Disk read-only")):

         # Double fault must log critical failure and re-raise original simulation exception:
         with pytest.raises(ValueError, match="Divergent field value"):
             run_simulation(
                 input_output_folder=tmp_path,
                 input_file_name=input_file.name,
                 output_file_name="output_manifest.json",
             )"""

replacement_204 = """    with patch("src.main.BASE_DIR", tmp_path), \\
         patch("src.main.load_and_validate_inputs", return_value=({}, {})), \\
         patch("src.main.SolverState", return_value=mock_state), \\
         patch("src.main.step_simulation", side_effect=ValueError("Divergent field value")), \\
         patch("src.main.archive_simulation_results", side_effect=OSError("Disk read-only")), \\
         pytest.raises(ValueError, match="Divergent field value"):
         # Double fault must log critical failure and re-raise original simulation exception:
         run_simulation(
             input_output_folder=tmp_path,
             input_file_name=input_file.name,
             output_file_name="output_manifest.json",
         )"""

updated = content
if target_170 in updated:
    updated = updated.replace(target_170, replacement_170)
    print("Successfully repaired occurrence at line 170.")
else:
    print("Notice: target_170 exact string not matched (may already be repaired).")

if target_204 in updated:
    updated = updated.replace(target_204, replacement_204)
    print("Successfully repaired occurrence at line 204.")
else:
    print("Notice: target_204 exact string not matched (may already be repaired).")

with open(path, "w", encoding="utf-8") as f:
    f.write(updated)
'



echo ""
echo "=============================================================================="
echo "Forensic audit and diagnostic sequence completed."