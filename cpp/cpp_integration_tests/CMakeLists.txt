cmake_minimum_required(VERSION 3.14)
project(NavierStokesIntegrationTests)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenMP REQUIRED)

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# ---------------------------------------------------------------------------
# Core Navier-Stokes Library Target (Ensuring ABI & Macro Consistency)
# ---------------------------------------------------------------------------

file(GLOB_RECURSE CORE_SOURCES "../src/*.cpp")

add_library(navier_stokes_core STATIC ${CORE_SOURCES})

target_include_directories(navier_stokes_core PUBLIC ../include)

# Propagate debug snapshot macros PUBLICLY so library and tests share identical object layout
target_compile_definitions(navier_stokes_core PUBLIC
    NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
)

target_link_libraries(navier_stokes_core
    PRIVATE
        OpenMP::OpenMP_CXX
)

# ---------------------------------------------------------------------------
# Full Pipeline Literate Integration Test
# ---------------------------------------------------------------------------

add_executable(test_full_pipeline_literate
    test_full_pipeline_literate.cpp
)

target_link_libraries(test_full_pipeline_literate
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
)

add_test(NAME FullPipelineLiterateTest COMMAND test_full_pipeline_literate)

# ---------------------------------------------------------------------------
# Mass Continuity & Divergence Integration Test
# ---------------------------------------------------------------------------

add_executable(test_mass_continuity
    test_mass_continuity.cpp
)

target_link_libraries(test_mass_continuity
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
)

add_test(NAME MassContinuityTest COMMAND test_mass_continuity)