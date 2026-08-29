#!/usr/bin/env bash
set -uo pipefail

BUILD_DIR="./build"
TEST_BIN="${BUILD_DIR}/test_full_pipeline_accelerated_flow"

echo "Listing all tests in suite..."
"${TEST_BIN}" --gtest_list_tests

# Run binary with gtest repetition or list order to isolate the predecessor
echo "Running full binary debug dump..."
"${TEST_BIN}" --gtest_break_on_failure