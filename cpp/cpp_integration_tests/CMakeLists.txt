cmake_minimum_required(VERSION 3.14)
project(NavierStokesIntegrationTests)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenMP REQUIRED)

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# --- No-Penetration Boundary Clamping Integration Test ---

add_executable(test_no_penetration_clamping
    test_no_penetration_clamping.cpp
)

target_link_libraries(test_no_penetration_clamping
    PRIVATE
        GTest::gtest_main
        navier_stokes_core
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
)

add_test(NAME NoPenetrationClampingTest COMMAND test_no_penetration_clamping)