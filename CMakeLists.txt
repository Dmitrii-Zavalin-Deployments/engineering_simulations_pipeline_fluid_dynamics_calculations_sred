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