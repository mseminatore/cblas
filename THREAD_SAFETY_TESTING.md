# Thread Safety Testing

## Overview

The `test_concurrent.c` test suite provides comprehensive thread safety testing for the CBLAS library. It validates that BLAS operations can be safely called from multiple threads simultaneously and detects potential race conditions.

## Test Coverage

### 1. Concurrent BLAS Operations
Tests multiple threads calling different Level 1 BLAS operations simultaneously:
- `cblas_scopy` - Vector copy
- `cblas_sdot` - Dot product
- `cblas_saxpy` - Vector addition
- `cblas_sscal` - Vector scaling
- `cblas_sasum` - Sum of absolute values
- `cblas_snrm2` - Euclidean norm

Each test runs with 2, 4, 8, and 16 concurrent threads to stress-test thread safety.

### 2. Concurrent Matrix Operations
Tests Level 2 and Level 3 BLAS operations with concurrent threads:
- `cblas_sgemm` - Matrix-matrix multiplication
- `cblas_sger` - Outer product
- `cblas_sgemv` - Matrix-vector multiplication

### 3. Thread Management
Tests concurrent calls to:
- `cblas_set_num_threads()` - Dynamic thread count adjustment
- Multiple threads changing thread pool size simultaneously

### 4. Initialization/Shutdown Cycles
Tests repeated cycles of:
- `cblas_init()` - Library initialization
- BLAS operations
- `cblas_shutdown()` - Library cleanup

## Running the Tests

### Standard Execution

```bash
# Using Make
make test_concurrent
./test_concurrent

# Using CMake
mkdir build && cd build
cmake ..
cmake --build . --target test_concurrent
./test_concurrent

# Using CTest
ctest -R test_concurrent -V
```

### Running with ThreadSanitizer (TSAN)

ThreadSanitizer is a powerful tool for detecting data races and threading issues.

#### Option 1: Using Make

```bash
make clean
CFLAGS="-fsanitize=thread -g -O1" make test_concurrent
./test_concurrent
```

#### Option 2: Using CMake

```bash
rm -rf build && mkdir build && cd build
cmake -DCMAKE_C_FLAGS="-fsanitize=thread -g -O1" ..
cmake --build . --target test_concurrent
./test_concurrent
```

## Interpreting Results

### Normal Execution
When running without ThreadSanitizer, the tests perform functional validation:
- ✓ Indicates successful test completion
- No errors detected means threads executed correctly without data corruption

### ThreadSanitizer Execution
When running with TSAN, additional race condition detection occurs:

```
WARNING: ThreadSanitizer: data race
```

TSAN warnings indicate potential thread safety issues in the library code. The test output includes:
- Location of the race (file and line number)
- Thread IDs involved
- Memory addresses accessed
- Call stacks for both racing accesses

## Known Issues

ThreadSanitizer has detected the following race conditions in the library:

1. **Statistics Initialization** (`util.c:76`)
   - Race in `stats_initialized` flag check
   - Multiple threads initializing stats table simultaneously

2. **GEMM Packed Buffers** (`gemm.c:360-384`)
   - Race conditions in `packedA` and `packedB` global buffers
   - Multiple threads packing matrices into shared static buffers

These are pre-existing issues in the library that should be addressed to ensure full thread safety.

## Test Configuration

Default test parameters (can be modified in `test_concurrent.c`):
```c
#define VECTOR_SIZE 1000              // Size of test vectors
#define MATRIX_SIZE 100               // Size of test matrices
#define ITERATIONS_PER_THREAD 100     // Iterations per thread
```

## Adding New Tests

To add new thread safety tests:

1. Create a new thread function following the pattern:
```c
void* test_new_operation_thread(void* arg)
{
    thread_args_t* args = (thread_args_t*)arg;
    // Allocate thread-local data
    // Perform BLAS operations
    // Report errors via args->error_count
    return NULL;
}
```

2. Add the test to `test_main()`:
```c
test_concurrent_new_operation(2);
test_concurrent_new_operation(4);
```

## Continuous Integration

The test suite can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Build and test with ThreadSanitizer
  run: |
    mkdir build && cd build
    cmake -DCMAKE_C_FLAGS="-fsanitize=thread -g" ..
    cmake --build .
    ctest --output-on-failure
```

## Requirements

- POSIX threads (pthread) support
- For ThreadSanitizer: GCC 4.8+, Clang 3.2+, or compatible compiler
- Sufficient memory for concurrent thread execution

## Performance Considerations

ThreadSanitizer adds significant overhead (5-15x slowdown). The tests may take longer to execute with TSAN enabled but provide comprehensive race condition detection.

## Future Enhancements

Potential additions to the test suite:
- Double-precision operations (d* functions)
- Complex number operations (c* functions)
- Stress testing with higher thread counts
- Memory leak detection with Valgrind
- Testing with different MT threshold values
