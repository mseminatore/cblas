# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

CBLAS is an experimental, performance-focused C implementation of a subset of BLAS (Basic Linear Algebra Subprograms). It intentionally excludes complex number support and is not a complete BLAS library — it's an optimization playground focused on SIMD, multi-threading, and cache-aware algorithms.

## Build Commands

```bash
# Makefile (Unix/Linux/macOS)
make                         # Build library + all executables
make test                    # Run ./blas_test
make install                 # Install to /opt/cblas
make CBLAS_ENABLE_MT=0       # Build without multi-threading

# CMake (all platforms)
mkdir build && cd build
cmake .. -DCBLAS_ENABLE_MT=ON
cmake --build . --config Release
ctest                        # Run full test suite
ctest --verbose              # Show test output
```

## Running Tests

```bash
# Individual test executables (from build dir or repo root after make)
./blas_test              # Main unit tests
./blas_stress            # Heavy stress tests (~2+ min)
./test_strided           # Strided vector tests
./test_concurrent        # Multi-threading tests
./test_level2_mt         # Level-2 MT tests
./test_autotune          # Auto-tuning tests
./test_gemm_accuracy     # GEMM numerical accuracy
```

## CMake Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `CBLAS_ENABLE_MT` | ON | Enable multi-threading |
| `CBLAS_CHECK_INPUTS` | ON | Input validation / `XERBLA` error handling |
| `CBLAS_USE_STATIC_BUFFERS` | ON | Static vs stack allocation for packed matrices |
| `CBLAS_MAX_THREADS` | 64 | Maximum supported threads |

`cblas_config.h` is auto-generated from `cblas_config.h.in` — never edit it directly.

## Architecture

### Three-Tier BLAS Hierarchy

- **Level 1** (`dot.c`, `axpy.c`, `scal.c`, etc.) — vector-vector ops; each in its own `.c` file
- **Level 2** (`ger.c`, `gemv.c`) — matrix-vector ops with cache blocking
- **Level 3** (`gemm.c`) — matrix-matrix ops; primary optimization target using 3-level GotoBLAS-style cache tiling with packed buffers (`packedA`, `packedB`)

Use `dot.c` as the canonical reference for Level-1 implementation patterns.

### Kernel Dispatch

`util.c` populates a global `blas_kernels` struct at startup with the best kernel for the detected CPU. `cpuid_x64.c` / `cpuid_arm64.c` detect CPU features (SSE, AVX2, FMA, AVX-512, NEON). Kernels live in `kernels/` and follow the naming pattern `<op>_k.c`, `<op>_k_avx.c`, `<op>_k_fma.c`, `<op>_k_neon.c`, etc.

### Threading

- `server.c` (POSIX) and `server_win32.c` (Windows) manage a persistent worker thread pool
- Work is submitted via `work_queue_t` structs; parameters are bundled in `cblas_args_t`
- Operations activate MT only above size thresholds; defaults are in `cblas.h` (e.g., `CBLAS_MT_DOT=500000`)
- Initialize with `cblas_init(n)`, shut down with `cblas_shutdown()`
- `platform/threading.h` provides the cross-platform mutex/condvar abstraction

### Performance Techniques

- **`_k_noinc` variants** — specialized kernels for `incx == 1 && incy == 1` (no index multiply overhead)
- **4-way loop unrolling** with separate accumulators (`sum0`–`sum3`) to reduce data dependencies
- **Prefetch macros** — configurable via `CBLAS_PREFETCH_THRESHOLD` / `CBLAS_PREFETCH_DISTANCE` in `cblas.h`
- **Auto-tuning** — `CBLAS_AUTO_TUNE=1` env var runs calibration benchmarks at init; also callable as `cblas_autotune_thresholds()`

## Code Conventions

**Function naming:** `cblas_` prefix, `s`/`d` precision, e.g., `cblas_sdot()`, `cblas_dgemm()`. Internal kernels: `cblas_sdot_k()`, `cblas_sdot_k_noinc()`.

**SIMD guards:** Use `#if defined(__x86_64__) || defined(_M_X64)` for x86 paths; `#ifdef __aarch64__` for ARM. Include `cblas_simd.h` only in kernel files that use SIMD intrinsics.

**Error handling:** Wrap with `CBLAS_CHECK_INPUTS` guard, use `CBLAS_VALIDATE_VEC1` / `CBLAS_VALIDATE_VEC2` macros, call `XERBLA(param_num)` on bad input.

**Stats:** Wrap with `CBLAS_STATS_START()` / `CBLAS_STATS_END(op, n, mt)` when `CBLAS_ENABLE_STATS` is defined.

**Matrix access:** `A(col, row)` macro with leading-dimension (`lda`) parameter handles non-contiguous submatrix storage.

## Adding a New Operation

1. Create `<op>.c` — implement public API with `CBLAS_VALIDATE_*` macros, MT dispatch, and a `_k_noinc` fast path
2. Add SIMD kernels in `kernels/<op>_k*.c` — guard each with platform detection
3. Register kernels in `util.c` (`blas_kernels` struct population)
4. Declare public function in `cblas.h` under the correct BLAS level section
5. Update `CMakeLists.txt` source list and `Makefile` `OBJS` variable
6. Add per-file SIMD compiler flags in `CMakeLists.txt` (see existing `set_source_files_properties` calls)
7. Add tests in `tests/test.c` using `TEST_ASSERT(expr, __FILE__, __LINE__)`
