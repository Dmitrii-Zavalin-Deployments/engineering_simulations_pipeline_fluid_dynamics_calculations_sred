cmake_minimum_required(VERSION 3.15)
project(NavierStokesSolver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(ENABLE_COVERAGE "Enable gcov coverage flags" OFF)

if(ENABLE_COVERAGE)
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

# Find required packages
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

enable_testing()

# Global include directory for all C++ targets
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)

# ---------------------------------------------------------------------------
# 1. Core Navier-Stokes Library Target
# ---------------------------------------------------------------------------
file(GLOB_RECURSE CORE_SOURCES "cpp/src/*.cpp")

# Exclude python_gate.cpp from the core C++ library (it's reserved strictly for python bindings)
list(FILTER CORE_SOURCES EXCLUDE REGEX "python_gate\\.cpp$")

add_library(navier_stokes_core STATIC ${CORE_SOURCES})

# Ensure position-independent code so it can be safely linked into the shared pybind module
set_target_properties(navier_stokes_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(navier_stokes_core PUBLIC
    NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
)

target_link_libraries(navier_stokes_core
    PUBLIC
        OpenMP::OpenMP_CXX
)

# ---------------------------------------------------------------------------
# 2. C++ Unit Tests Target
# ---------------------------------------------------------------------------
file(GLOB UNIT_TEST_SOURCES "cpp/cpp_unit_tests/*.cpp")
add_executable(unit_tests ${UNIT_TEST_SOURCES})

target_link_libraries(unit_tests
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)

add_test(NAME UnitTests COMMAND unit_tests)

# ---------------------------------------------------------------------------
# 3. C++ Integration Tests Targets
# ---------------------------------------------------------------------------
# Copy input data required by integration tests into the build tree
file(COPY cpp/cpp_integration_tests/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Full Pipeline Literate Test
add_executable(test_full_pipeline_literate
    cpp/cpp_integration_tests/test_full_pipeline_literate.cpp
)
target_link_libraries(test_full_pipeline_literate
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME FullPipelineLiterateTest COMMAND test_full_pipeline_literate)

# Mass Continuity & Divergence Test
add_executable(test_mass_continuity
    cpp/cpp_integration_tests/test_mass_continuity.cpp
)
target_link_libraries(test_mass_continuity
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME MassContinuityTest COMMAND test_mass_continuity)

# ---------------------------------------------------------------------------
# 4. Python Pybind11 Extension Module Target (`navier_stokes_cpp`)
# ---------------------------------------------------------------------------
file(GLOB ALL_CPP_SOURCES "cpp/src/*.cpp") # Includes python_gate.cpp

pybind11_add_module(navier_stokes_cpp MODULE ${ALL_CPP_SOURCES})
target_link_libraries(navier_stokes_cpp 
    PRIVATE 
        navier_stokes_core 
        OpenMP::OpenMP_CXX
)
