#!/usr/bin/env bash
set -euo pipefail

echo "========================================================================"
echo "APPLYING FIX: Correcting Face Flux Overwrites in orchestrator.cpp"
echo "========================================================================"

# Replace direct assignment (=) with accumulation (+=) for east, north, and top face fluxes
sed -i 's/dudx = u_face\[idx_e_face\] \/ dims_\.dx;/dudx += u_face[idx_e_face] \/ dims_.dx;/g' cpp/src/orchestrator.cpp
sed -i 's/dvdy = v_face\[idx_n_face\] \/ dims_\.dy;/dvdy += v_face[idx_n_face] \/ dims_.dy;/g' cpp/src/orchestrator.cpp
sed -i 's/dwdz = w_face\[idx_t_face\] \/ dims_\.dz;/dwdz += w_face[idx_t_face] \/ dims_.dz;/g' cpp/src/orchestrator.cpp

echo "Fix applied successfully. Rebuilding project..."