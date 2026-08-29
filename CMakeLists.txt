cmake_minimum_required(VERSION 3.15)
project(NavierStokesSolver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(ENABLE_COVERAGE "Enable gcov coverage flags" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(ENABLE_COVERAGE)
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

if(ENABLE_ASAN)
    message(STATUS "Building with AddressSanitizer and UndefinedBehaviorSanitizer")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
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

# Exclude python_gate.cpp from the core C++ library (reserved strictly for python bindings)
list(FILTER CORE_SOURCES EXCLUDE REGEX "python_gate\\.cpp$")

add_library(navier_stokes_core STATIC ${CORE_SOURCES})

# Ensure position-independent code so it can be safely linked into the shared pybind module
set_target_properties(navier_stokes_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

# Acceleration with Gravity Integration Test
add_executable(test_full_pipeline_accelerated_with_gravity
    cpp/cpp_integration_tests/test_full_pipeline_accelerated_with_gravity.cpp
)
target_link_libraries(test_full_pipeline_accelerated_with_gravity
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME FullPipelineAcceleratedWithGravityTest COMMAND test_full_pipeline_accelerated_with_gravity)

# Accelerated Flow Integration Test
add_executable(test_full_pipeline_accelerated_flow
    cpp/cpp_integration_tests/test_full_pipeline_accelerated_flow.cpp
)
target_link_libraries(test_full_pipeline_accelerated_flow
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME FullPipelineAcceleratedFlowTest COMMAND test_full_pipeline_accelerated_flow)

# Full Pipeline Literate Test
add_executable(test_full_pipeline_constant_flow
    cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp
)
target_link_libraries(test_full_pipeline_constant_flow
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME FullPipelineLiterateTest COMMAND test_full_pipeline_constant_flow)

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

# Plane Poiseuille Flow Integration Test
add_executable(test_plane_poiseuille_flow
    cpp/cpp_integration_tests/test_plane_poiseuille_flow.cpp
)
target_link_libraries(test_plane_poiseuille_flow
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME PlanePoiseuilleFlowTest COMMAND test_plane_poiseuille_flow)

# CFL & Stability Integration Test
add_executable(test_cfl_and_stability
    cpp/cpp_integration_tests/test_cfl_and_stability.cpp
)
target_link_libraries(test_cfl_and_stability
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME CflAndStabilityTest COMMAND test_cfl_and_stability)

# Free-Slip Symmetry Plane Boundary Verification Test
add_executable(test_free_slip_symmetry
    cpp/cpp_integration_tests/test_free_slip_symmetry.cpp
)
target_link_libraries(test_free_slip_symmetry
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME FreeSlipSymmetryTest COMMAND test_free_slip_symmetry)

# No-Penetration Clamping Integration Test
add_executable(test_no_penetration_clamping
    cpp/cpp_integration_tests/test_no_penetration_clamping.cpp
)
target_link_libraries(test_no_penetration_clamping
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME NoPenetrationClampingTest COMMAND test_no_penetration_clamping)

# No-Slip Wall Shear Integration Test
add_executable(test_no_slip_wall_shear
    cpp/cpp_integration_tests/test_no_slip_wall_shear.cpp
)
target_link_libraries(test_no_slip_wall_shear
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME NoSlipWallShearTest COMMAND test_no_slip_wall_shear)

# Pressure-Driven Channel Flow Integration Test
add_executable(test_pressure_driven_channel
    cpp/cpp_integration_tests/test_pressure_driven_channel.cpp
)
target_link_libraries(test_pressure_driven_channel
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME PressureDrivenChannelTest COMMAND test_pressure_driven_channel)

# Internal Solid Object Masking Integration Test
add_executable(test_solid_masking
    cpp/cpp_integration_tests/test_solid_masking.cpp
)
target_link_libraries(test_solid_masking
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME SolidMaskingTest COMMAND test_solid_masking)

# Static Pool Equilibrium & Spurious Current Suppression Integration Test
add_executable(test_static_pool_equilibrium
    cpp/cpp_integration_tests/test_static_pool_equilibrium.cpp
)
target_link_libraries(test_static_pool_equilibrium
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME StaticPoolEquilibriumTest COMMAND test_static_pool_equilibrium)

# Velocity Inlet / Pressure Outlet Integration Test
add_executable(test_velocity_inlet_outlet
    cpp/cpp_integration_tests/test_velocity_inlet_outlet.cpp
)
target_link_libraries(test_velocity_inlet_outlet
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME VelocityInletOutletTest COMMAND test_velocity_inlet_outlet)

# Waterfall Inflow Dynamics Integration Test
add_executable(test_waterfall_inflow
    cpp/cpp_integration_tests/test_waterfall_inflow.cpp
)
target_link_libraries(test_waterfall_inflow
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        Threads::Threads
)
add_test(NAME WaterfallInflowTest COMMAND test_waterfall_inflow)

# ---------------------------------------------------------------------------
# 4. Python Pybind11 Extension Module Target (`navier_stokes_cpp`)
# ---------------------------------------------------------------------------
file(GLOB ALL_CPP_SOURCES "cpp/src/*.cpp") # Includes python_gate.cpp

pybind11_add_module(navier_stokes_cpp MODULE ${ALL_CPP_SOURCES})

# Force the compiled module output directly into the repository root directory
# so that Python's sys.path can discover and import it natively.
set_target_properties(navier_stokes_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(navier_stokes_cpp 
    PRIVATE 
        navier_stokes_core 
        OpenMP::OpenMP_CXX
)