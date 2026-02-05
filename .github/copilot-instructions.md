# CBLAS Copilot Instructions

This is an experimental, performance-focused subset of the BLAS (Basic Linear Algebra Subprograms) library in C. The codebase emphasizes deep optimization with SIMD, multi-threading, and cache-aware algorithms.

## Architecture Overview

**Three-tier BLAS hierarchy:**
- Level 1: Vector-vector ops (dot, axpy, scal, copy, swap, asum, nrm2, rot, rotg) in individual .c files
- Level 2: Matrix-vector ops (ger, gemv) with cache blocking  
- Level 3: Matrix-matrix ops (gemm) - primary optimization target

**Threading architecture:**
- Platform-specific thread servers: [server.c](server.c) (pthread) and [server_win32.c](server_win32.c) (Windows)
- Work queue system with `work_queue_t` structures dispatched to worker threads
- Kernels execute via `cblas_args_t` parameter bundles through function pointers
- MT thresholds in [cblas.h](cblas.h): `CBLAS_MT_DOT=100000`, `CBLAS_MT_GEMM=10000`, etc. control when threading activates
- Thread management: `cblas_init()`, `cblas_set_num_threads()`, `cblas_shutdown()`
- Debug tracing: Enable with `MT_DEBUG` macro, JSON output with `MT_DEBUG_JSON`

**Performance strategy:**
- Kernel specialization: Most operations have optimized `_k_noinc` variants when `incx == 1 && incy == 1`
- Loop unrolling: Inner loops use 4-way unrolling with separate accumulators (e.g., `sum0`, `sum1`, `sum2`, `sum3`) to reduce dependencies
- SIMD paths conditionally compiled via platform detection (e.g., `__x86_64__`, `__aarch64__`) and `USE_INTEL_FMA` preprocessor flags
- Cache blocking: [gemm.c](gemm.c) uses runtime-determined tile sizes (`mc`, `kc`, `nb`) with packed buffers (`packedA`, `packedB`)
- Platform detection: [cpuid_x64.c](cpuid_x64.c) and [cpuid_arm64.c](cpuid_arm64.c) detect CPU features at runtime
- Kernel dispatch: [util.c](util.c) populates `blas_kernels` struct with CPU-specific optimized kernels based on `cpu_get_features()` flags (CPU_SSE, CPU_AVX2, CPU_NEON, etc.)
- Prefetching: Configurable via `CBLAS_PREFETCH_THRESHOLD` and `CBLAS_PREFETCH_DISTANCE` macros

## Build System

**Cross-platform builds:**
```bash
# Make (Unix/Linux/macOS)
make                    # Builds library + all executables
make test               # Alias for running ./blas_test
make install            # Install to /opt/cblas

# CMake (all platforms)
mkdir build && cd build
cmake .. -DCBLAS_ENABLE_MT=ON
cmake --build . --config Release
ctest                   # Run full test suite
```

**Configuration options** (CMake and Makefile):
- `CBLAS_ENABLE_MT` (default: ON/1) - Enable multi-threading
- `CBLAS_CHECK_INPUTS` (default: ON/1) - Enable input validation and `XERBLA` error handling
- `CBLAS_USE_STATIC_BUFFERS` (default: ON/1) - Use static vs stack allocation for packed matrices
- `CBLAS_MAX_THREADS` (default: 64) - Maximum supported threads

CMake auto-selects platform files: [CMakeLists.txt](CMakeLists.txt#L79-L85) chooses server_win32.c vs server.c and cpuid variant based on platform/arch.

**Compiler flags:** Architecture-specific SIMD enabled via `-mavx2 -mfma` (x64 GCC/Clang) or `/arch:AVX` (MSVC).

**Generated config:** [cblas_config.h](cblas_config.h) is auto-generated from [cblas_config.h.in](cblas_config.h.in) with build-time settings - never edit directly.

## Code Conventions

**Header organization:**
- [cblas.h](cblas.h) - Main API header, type definitions, configuration macros
- [cblas_simd.h](cblas_simd.h) - Platform-specific SIMD intrinsics (immintrin.h, arm_neon.h)
- Include `cblas_simd.h` only in kernel files that implement SIMD optimizations

**Function naming:**
- Prefix: `cblas_` for public API
- Precision: `s` = single (float), `d` = double  
- Examples: `cblas_sdot()`, `cblas_dgemm()`
- Internal kernels: `cblas_sdot_k()`, `cblas_sdot_k_noinc()`

**Matrix/vector access macros:**
```c
#define A(col, row) a[((row) * lda + (col))]  // In gemm.c
#define Y(i) y[(i) * incx]                     // Common pattern
```
Leading dimension (`lda`, `ldb`, `ldc`) handles non-contiguous matrix storage for submatrices.

**Error handling:**
- Guarded by `CBLAS_CHECK_INPUTS` and `CBLAS_XERBLA_INPUTS` preprocessor flags
- Use `XERBLA(param_num)` macro which calls `xerbla()` in [util.c](util.c)
- Early return with default values on invalid input (e.g., `sum = 0.0f` for dot product)
- Input validation macros: `CBLAS_VALIDATE_VEC1`, `CBLAS_VALIDATE_VEC2` reduce code duplication

**Configuration switches in [cblas.h](cblas.h#L44-L72):**
- `MT_ENABLED`: Enable multi-threading (set by `CBLAS_ENABLE_MT`)
- `USE_STATIC_BUFFERS`: Static vs stack allocation for packed matrices
- `MT_DEBUG`: Trace threading with `MT_TRACE()` macro
- `MT_DEBUG_JSON`: Output debug traces as JSON
- `CBLAS_ENABLE_STATS`: Performance counter tracking
- Always check these flags before adding performance features

## Testing

**Test structure:**
- Framework: [test.h](test.h) with colored terminal output macros (`CHECK_MARK ✓`, `X_MARK ❌`)
- Unit tests: [test.c](test.c) uses `TEST_ASSERT(expr, __FILE__, __LINE__)` macro, tracks failures in `test_failures` counter
- Stress tests: [test_stress.c](test_stress.c) for heavy workloads
- MT tests: [test_concurrent.c](test_concurrent.c), [test_level2_mt.c](test_level2_mt.c), [test_mt_debug.c](test_mt_debug.c)
- Performance: [gemm_perf.c](gemm_perf.c), [ger_perf.c](ger_perf.c), [dot_perf.c](dot_perf.c) report GFlops
- Threshold tuning: [dot_threshold_tuning.c](dot_threshold_tuning.c), [test_threshold.c](test_threshold.c)

**Test patterns:**
```c
// From test.c - uses predefined test vectors (sa, sb, szeros, sones)
cblas_scopy(ARRAY_SIZE(sones), sr, 1, sones, 1);
TEST_ASSERT(equal_sarray_epsilon(sr, sones, ARRAY_SIZE(sones)), __FILE__, __LINE__);
```

**Run tests:**
- Make: `./blas_test` or individual test executables
- CMake: `ctest` or `cmake --build . && ctest`

## Key Integration Points

**Initialization:**
```c
cblas_init(CBLAS_DEFAULT_THREADS);  // Required before any BLAS calls
cblas_set_num_threads(n);           // Dynamic thread count adjustment
cblas_shutdown();                    // Cleanup thread pool
```

**CPU feature detection:** [util.c](util.c) populates `blas_kernels` struct with optimized kernel pointers based on `cpu_get_features()` flags (CPU_SSE, CPU_AVX2, CPU_FMA, CPU_NEON, etc.).

**Performance statistics:** When `CBLAS_ENABLE_STATS` is defined:
- `cblas_record_operation(op, n, mt_used, time_sec)` - Record operation
- `cblas_get_stats(op)` - Retrieve stats for operation
- Stats include: `total_calls`, `total_elements`, `mt_activations`, `total_time_sec`

## Adding New Operations

1. Create `<operation>.c` with kernel function(s) accepting `cblas_args_t*`
2. Include `#include "cblas.h"` and `#include "cblas_simd.h"` (if using SIMD)
3. Add public API function with error checking using `CBLAS_VALIDATE_VEC1/VEC2` macros
4. Implement optimized `_k_noinc` variant for `inc==1` cases
5. Add SIMD code path guarded by platform detection (e.g., `#if defined(__x86_64__) || defined(_M_X64)`)
6. Update [CMakeLists.txt](CMakeLists.txt) and [Makefile](Makefile) source lists (OBJS variable)
7. Add tests to [test.c](test.c) and optionally perf harness
8. Export function declaration in [cblas.h](cblas.h) under appropriate BLAS level section (lines 1100-1250)
9. Add stats tracking with `CBLAS_STATS_START()` / `CBLAS_STATS_END()` macros

Refer to [dot.c](dot.c#L1-L100) for canonical Level-1 implementation pattern with SIMD optimization.
