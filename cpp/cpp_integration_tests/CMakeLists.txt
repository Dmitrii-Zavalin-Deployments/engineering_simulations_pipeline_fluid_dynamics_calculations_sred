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

add_executable(test_hydrostatic_decoupling
    test_hydrostatic_decoupling.cpp
)

target_link_libraries(test_hydrostatic_decoupling
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME HydrostaticDecouplingTest COMMAND test_hydrostatic_decoupling)
