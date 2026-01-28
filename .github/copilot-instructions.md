# CBLAS Copilot Instructions

This is an experimental, performance-focused subset of the BLAS (Basic Linear Algebra Subprograms) library in C. The codebase emphasizes deep optimization with SIMD, multi-threading, and cache-aware algorithms.

## Architecture Overview

**Three-tier BLAS hierarchy:**
- Level 1: Vector-vector ops (dot, axpy, scal, copy, swap, etc.) in individual .c files
- Level 2: Matrix-vector ops (ger, gemv)  
- Level 3: Matrix-matrix ops (gemm) - primary optimization target

**Threading architecture:**
- Platform-specific thread servers: [server.c](server.c) (pthread) and [server_win32.c](server_win32.c) (Windows)
- Work queue system with `work_queue_t` structures dispatched to worker threads
- Kernels execute via `cblas_args_t` parameter bundles through function pointers
- MT thresholds in [cblas.h](cblas.h): `CBLAS_MT_DOT`, `CBLAS_MT_GEMM`, etc. control when threading activates

**Performance strategy:**
- Kernel specialization: Most operations have optimized `_k_noinc` variants when `incx == 1 && incy == 1`
- SIMD paths conditionally compiled via `USE_SSE`, `USE_SIMD`, `USE_INTEL_FMA` preprocessor flags
- Cache blocking: gemm uses `mc`, `kc`, `nb` tile sizes with packed buffers (`packedA`, `packedB`)
- Platform detection: [cpuid_x64.c](cpuid_x64.c) and [cpuid_arm64.c](cpuid_arm64.c) detect CPU features

## Build System

**Cross-platform builds:**
```bash
# Make (Unix/Linux/macOS)
make                    # Builds library + all executables
make test               # Run blas_test suite
make install            # Install to /opt/cblas

# CMake (all platforms)
mkdir build && cd build
cmake ..
cmake --build . --config Release
ctest                   # Run tests
```

CMake auto-selects platform files: [CMakeLists.txt](CMakeLists.txt) lines 32-44 choose server_win32.c vs server.c and cpuid variant based on platform/arch.

**Compiler flags:** Architecture-specific SIMD enabled via `-mavx2 -mfma` (x64) or `/arch:AVX` (MSVC).

## Code Conventions

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

**Error handling:**
- Guarded by `CBLAS_CHECK_INPUTS` and `CBLAS_XERBLA_INPUTS` preprocessor flags
- Use `XERBLA(param_num)` macro which calls `xerbla()` in [util.c](util.c)
- Early return with default values on invalid input

**Configuration switches in [cblas.h](cblas.h) lines 44-72:**
- `MT_ENABLED`: Enable multi-threading
- `USE_STATIC_BUFFERS`: Static vs stack allocation for packed matrices
- `MT_DEBUG`: Trace threading with `MT_TRACE()` macro
- Always check these flags before adding performance features

## Testing

**Test structure:**
- Framework: [test.h](test.h) with colored terminal output macros (`CHECK_MARK`, `X_MARK`)
- Unit tests: [test.c](test.c) uses `TEST_ASSERT()` macro, tracks failures in `test_failures` counter
- Stress tests: [test_stress.c](test_stress.c) for heavy workloads
- Performance: [gemm_perf.c](gemm_perf.c), [ger_perf.c](ger_perf.c), [dot_perf.c](dot_perf.c) report GFlops

**Test patterns:**
```c
// From test.c - uses predefined test vectors (sa, sb, szeros, sones)
cblas_scopy(ARRAY_SIZE(sones), sr, 1, sones, 1);
TEST_ASSERT(equal_sarray_epsilon(sr, sones, ARRAY_SIZE(sones)), __FILE__, __LINE__);
```

Run: `./blas_test` or `cmake --build . && ctest`

## Key Integration Points

**Initialization:**
```c
cblas_init(CBLAS_DEFAULT_THREADS);  // Required before any BLAS calls
cblas_set_num_threads(n);           // Dynamic thread count adjustment
cblas_shutdown();                    // Cleanup
```

**CPU feature detection:** [util.c](util.c) populates `blas_kernels` struct with optimized kernel pointers based on `cpu_get_features()` flags (CPU_SSE, CPU_AVX2, CPU_NEON, etc.).

## Adding New Operations

1. Create `<operation>.c` with kernel function(s) accepting `cblas_args_t*`
2. Add public API function with error checking
3. Implement optimized `_k_noinc` variant for `inc==1` cases
4. Add SIMD code path guarded by `#if defined(USE_SSE) && defined(USE_SIMD)`
5. Update [CMakeLists.txt](CMakeLists.txt) and [Makefile](Makefile) source lists
6. Add tests to [test.c](test.c) and optionally perf harness
7. Export function declaration in [cblas.h](cblas.h) under appropriate BLAS level section

Refer to [dot.c](dot.c) (lines 1-100) for canonical Level-1 implementation pattern.
