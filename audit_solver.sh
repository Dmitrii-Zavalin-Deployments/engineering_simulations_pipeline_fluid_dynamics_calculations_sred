#!/usr/bin/env bash
set -e

# Formatting colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BLUE}${BOLD}====================================================================${NC}"
echo -e "${BLUE}${BOLD} 🔍 NAVIER-STOKES SOLVER: DIAGNOSTIC & MULTI-THREADING AUDIT        ${NC}"
echo -e "${BLUE}${BOLD}====================================================================${NC}"

# Find all relevant C++ source and header files, ignoring build and external submodules
CPP_FILES=$(find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cc" \) \
    ! -path "*/build/*" ! -path "*/googletest/*" ! -path "*/.git/*")

if [ -z "$CPP_FILES" ]; then
    echo -e "${RED}Error: No C++ source files found in current directory.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}${BOLD}[1/5] LOCATING LAPLACIAN & GRADIENT COMPUTATION (Targeting [X:inf] Bug)${NC}"
echo -e "Searching for finite-difference division by dx, dy, dz, and Laplacian implementations..."
grep -rn -E "(Laplacian|laplacian|dx \* dx|dy \* dy|dz \* dz|du_dx|dv_dy|dw_dz)" $CPP_FILES || echo "None found."

echo -e "\n${YELLOW}${BOLD}[2/5] LOCATING BOUNDARY CONDITION & GHOST CELL APPLICATORS${NC}"
echo -e "Searching for boundary enforcement, mask updates, and ghost cell indexing..."
grep -rn -E "(BoundaryCondition|apply_boundary|ghost|mask|bc_list|x_min|x_max|y_min|y_max)" $CPP_FILES || echo "None found."

echo -e "\n${YELLOW}${BOLD}[3/5] LOCATING PRESSURE POISSON SOLVER KERNEL${NC}"
echo -e "Searching for iterative solver loops, Gauss-Seidel, Jacobi, or SOR implementations..."
grep -rn -E "(poisson|Poisson|max_poisson_iterations|poisson_tolerance|solve_pressure)" $CPP_FILES || echo "None found."

echo -e "\n${YELLOW}${BOLD}[4/5] OPENMP MULTI-THREADING AUDIT${NC}"
echo -e "Checking build configuration for OpenMP flags..."
if [ -f "CMakeLists.txt" ]; then
    echo -e "${BLUE}CMakeLists.txt OpenMP search:${NC}"
    grep -i "openmp" CMakeLists.txt || echo -e "${RED}⚠️ OpenMP not explicitly referenced in CMakeLists.txt!${NC}"
fi

echo -e "\nSearching C++ source code for '#pragma omp' loop decorations..."
OMP_MATCHES=$(grep -rn "#pragma omp" $CPP_FILES || true)
if [ -n "$OMP_MATCHES" ]; then
    echo -e "${GREEN}Found existing OpenMP pragmas:${NC}"
    echo "$OMP_MATCHES"
else
    echo -e "${RED}⚠️ No '#pragma omp' pragmas found in C++ source files! Compute loops are currently running single-threaded.${NC}"
fi

echo -e "\n${YELLOW}${BOLD}[5/5] IDENTIFYING UNPARALLELIZED COMPUTE LOOPS${NC}"
echo -e "Checking for candidates needing '#pragma omp parallel for':"
grep -rn -E "for\s*\(\s*int\s+[ijk]\s*=" $CPP_FILES | head -n 25 || echo "None found."

echo -e "\n${BLUE}${BOLD}====================================================================${NC}"
echo -e "${GREEN}${BOLD} Audit Complete! Review the file paths above to apply targeted fixes.${NC}"
echo -e "${BLUE}${BOLD}====================================================================${NC}"
