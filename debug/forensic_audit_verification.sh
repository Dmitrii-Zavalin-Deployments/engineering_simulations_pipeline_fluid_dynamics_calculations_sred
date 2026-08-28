#!/usr/bin/env bash
set -eu0px

# Patch test_full_pipeline_constant_flow.cpp to dump detailed vector values around index 81/82 at failure
TEST_FILE="cpp/cpp_integration_tests/test_full_pipeline_constant_flow.cpp"

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: ${TEST_FILE} not found from current directory."
    exit 1
fi

echo "Injecting detailed diagnostic print into ${TEST_FILE}..."

python3 -c "
file_path = '${TEST_FILE}'
with open(file_path, 'r') as f:
    content = f.read()

# Target the assertion block where u[idx] failure occurs
target_snippet = '''    EXPECT_NEAR(u[idx], 0.0, 1e-6)'''

replacement_snippet = '''    if (std::abs(u[idx]) > 1e-6) {
        std::fprintf(stderr, \"[DETAILED_DUMP] Failure at idx=%zu (i=%d, j=%d, k=%d): u=%g\\n\", idx, 2, 2, 1, u[idx]);
        // Print neighborhood u values
        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                size_t n_idx = (2 + di) + 8 * ((2 + dj) + 8 * 1);
                std::fprintf(stderr, \"  Neighbor u[%zu] = %g\\n\", n_idx, u[n_idx]);
            }
        }
    }
    EXPECT_NEAR(u[idx], 0.0, 1e-6)'''

if target_snippet in content and '[DETAILED_DUMP]' not in content:
    content = content.replace(target_snippet, replacement_snippet, 1)
    with open(file_path, 'w') as f:
        f.write(content)
    print('Successfully patched test file with neighborhood vector inspection.')
else:
    print('Target snippet not found or already patched.')
"

echo "Rebuilding and running test..."
cd build
cmake --build . --target test_full_pipeline_constant_flow -j$(nproc)
ctest -R test_full_pipeline_constant_flow --output-on-failure