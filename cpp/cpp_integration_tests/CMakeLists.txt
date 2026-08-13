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

# --- Scenario-based Boundary Condition Tests ---

add_executable(test_velocity_inlet_outlet
    test_velocity_inlet_outlet.cpp
)

target_link_libraries(test_velocity_inlet_outlet
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME VelocityInletOutletTest COMMAND test_velocity_inlet_outlet)


add_executable(test_pressure_driven_channel
    test_pressure_driven_channel.cpp
)

target_link_libraries(test_pressure_driven_channel
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME PressureDrivenChannelTest COMMAND test_pressure_driven_channel)


add_executable(test_no_slip_wall_shear
    test_no_slip_wall_shear.cpp
)

target_link_libraries(test_no_slip_wall_shear
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME NoSlipWallShearTest COMMAND test_no_slip_wall_shear)


add_executable(test_free_slip_symmetry
    test_free_slip_symmetry.cpp
)

target_link_libraries(test_free_slip_symmetry
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME FreeSlipSymmetryTest COMMAND test_free_slip_symmetry)


# --- Eulerian Cell Masking & Solid Wall Tests ---

add_executable(test_solid_masking
    test_solid_masking.cpp
)

target_link_libraries(test_solid_masking
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME SolidMaskingTest COMMAND test_solid_masking)


# --- Grid Staggering & Discretization Tests ---

add_executable(test_staggered_mac_grid
    test_staggered_mac_grid.cpp
)

target_link_libraries(test_staggered_mac_grid
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME StaggeredMacGridTest COMMAND test_staggered_mac_grid)


# --- Stability & CFL Condition Tests ---

add_executable(test_cfl_and_stability
    test_cfl_and_stability.cpp
)

target_link_libraries(test_cfl_and_stability
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME CflAndStabilityTest COMMAND test_cfl_and_stability)


# --- Canonical CFD Benchmarks ---

add_executable(test_canonical_flows
    test_canonical_flows.cpp
)

target_link_libraries(test_canonical_flows
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
)

add_test(NAME CanonicalFlowsTest COMMAND test_canonical_flows)
