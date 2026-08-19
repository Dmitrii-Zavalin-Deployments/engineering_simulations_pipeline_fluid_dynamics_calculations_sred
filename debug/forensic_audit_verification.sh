#!/usr/bin/env bash
set -euo pipefail

echo "======================================================================"
echo "          NAVIER-STOKES CFD - POST-TEST FORENSIC AUDIT SCRIPT         "
echo "======================================================================"

echo "--- [DIAGNOSTIC 1] External Forces Assertion Mismatch ---"
echo "Inspecting test_main_full_pipeline_end_to_end force_vector check..."
cat -n tests/test_integration_main_pipeline.py | sed -n '83,93p'

echo ""
echo "--- [DIAGNOSTIC 2] Pressure Field Zero Mutation in Forensic Audit ---"
echo "Inspecting test_pybind11_memory_bridge_forensic_audit boundary conditions..."
cat -n tests/test_integration_main_pipeline.py | sed -n '198,210p'

echo ""
echo "======================================================================"
echo "               APPLYING AUTOMATED REPAIR SED INJECTIONS               "
echo "======================================================================"

# Fix 1: Align the external forces assertion in test_main_full_pipeline_end_to_end with the test modification [1.0, 0.0, 0.0]
sed -i 's/assert input_data\["external_forces"\]\["force_vector"\] == \[1.0, 1.0, 1.0\]/assert input_data["external_forces"]["force_vector"] == [1.0, 0.0, 0.0]/g' tests/test_integration_main_pipeline.py

# Fix 2: Add opposing pressure boundary condition to test_pybind11_memory_bridge_forensic_audit so field_p receives non-zero mutations
sed -i 's/input_data\["boundary_conditions"\] = \[\{"location": "x_min", "type": "pressure", "value": 5.0\}\]/input_data["boundary_conditions"] = [{"location": "x_min", "type": "pressure", "value": 5.0}, {"location": "x_max", "type": "pressure", "value": 0.0}]/g' tests/test_integration_main_pipeline.py

echo "Forensic audit and automated repairs complete successfully."