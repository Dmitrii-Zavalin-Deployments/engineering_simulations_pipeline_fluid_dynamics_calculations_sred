cmake_minimum_required(VERSION 3.14)
project(NavierStokesIntegrationTests)

# Force the debug snapshot macro definition globally for test compilation units
add_definitions(-DNAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenMP REQUIRED)

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# ---------------------------------------------------------------------------
# Full Pipeline Literate Integration Test
# ---------------------------------------------------------------------------

add_executable(test_full_pipeline_literate
    test_full_pipeline_literate.cpp
)

target_compile_definitions(test_full_pipeline_literate
    PRIVATE
        NAVIER_STOKES_ORCHESTRATOR_DEBUG_DUMP_FIELDS
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