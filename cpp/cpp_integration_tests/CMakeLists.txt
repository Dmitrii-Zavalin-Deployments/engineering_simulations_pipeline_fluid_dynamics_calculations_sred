cmake_minimum_required(VERSION 3.14)
project(NavierStokesIntegrationTests)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_projection_pipeline
    test_projection_pipeline.cpp
)

target_link_libraries(test_projection_pipeline
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME ProjectionPipelineTest COMMAND test_projection_pipeline)

add_executable(test_static_pool_equilibrium
    test_static_pool_equilibrium.cpp
)

target_link_libraries(test_static_pool_equilibrium
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME StaticPoolDiagnosticTest COMMAND test_static_pool_equilibrium)

add_executable(test_waterfall_inflow
    test_waterfall_inflow.cpp
)

target_link_libraries(test_waterfall_inflow
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME WaterfallDiagnosticTest COMMAND test_waterfall_inflow)

add_executable(test_boundary_conditions
    test_boundary_conditions.cpp
)

target_link_libraries(test_boundary_conditions
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME BoundaryConditionsTest COMMAND test_boundary_conditions)
