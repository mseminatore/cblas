# Multi-Threading in CBLAS

This document describes which BLAS operations support multi-threading and their activation thresholds.

## Multi-Threaded Operations

| Level | Operation | File | Threshold Macro | Default Value | Activation Condition |
|-------|-----------|------|-----------------|---------------|----------------------|
| **Level 1** | `dot` (sdot/ddot) | dot.c | `CBLAS_MT_DOT` | 32768 | `n > threshold` |
| **Level 1** | `axpy` (saxpy/daxpy) | axpy.c | `CBLAS_MT_AXPY` | 32768 | `n > threshold` |
| **Level 1** | `copy` (scopy/dcopy) | copy.c | `CBLAS_MT_COPY` | 16384 | `n > threshold` |
| **Level 1** | `swap` (sswap/dswap) | swap.c | `CBLAS_MT_COPY` | 16384 | `n > threshold` |
| **Level 2** | `ger` (sger/dger) | ger.c | `CBLAS_MT_GER` | 2048 | `m*n > threshold` |
| **Level 2** | `gemv` (sgemv/dgemv) | gemv.c | `CBLAS_MT_GEMV` | 4096 | `m*n > threshold` |
| **Level 3** | `gemm` (sgemm/dgemm) | gemm.c | `CBLAS_MT_GEMM` | 4096 | `m*n*k > threshold` |

## Non-Multi-Threaded Operations

The following Level-1 operations are **not** multi-threaded due to low compute intensity:

- `scal` / `dscal` - Vector scaling
- `asum` / `dasum` - Sum of absolute values
- `nrm2` / `dnrm2` - Euclidean norm
- `rot` / `drot` - Givens rotation
- `rotg` / `drotg` - Givens rotation construction
- `setv` - Vector set

## Threshold Configuration

Thresholds are defined in `cblas.h` and can be adjusted at runtime via global variables in `util.c`:

```c
// Runtime-adjustable thresholds (util.c)
extern CBLAS_INDEX cblas_mt_dot_threshold;
extern CBLAS_INDEX cblas_mt_axpy_threshold;
extern CBLAS_INDEX cblas_mt_copy_threshold;
extern CBLAS_INDEX cblas_mt_ger_threshold;
extern CBLAS_INDEX cblas_mt_gemm_threshold;
extern CBLAS_INDEX cblas_mt_gemv_threshold;
```

## Thread Management

```c
cblas_init(CBLAS_DEFAULT_THREADS);  // Initialize thread pool
cblas_set_num_threads(n);           // Adjust thread count at runtime
cblas_shutdown();                    // Cleanup thread pool
```

## Implementation Details

- Thread dispatch uses `work_queue_t` structures defined in `cblas.h`
- Platform-specific implementations: `server.c` (pthread) and `server_win32.c` (Windows)
- Work is distributed via `cblas_execute()` or `cblas_execute_async()` functions
- Debug tracing available via `MT_DEBUG` macro (JSON output with `MT_DEBUG_JSON`)

## Tuning

Use the threshold tuning utilities to find optimal values for your hardware:

- `dot_threshold_tuning.c` - Tune `CBLAS_MT_DOT`
- `dot_threshold_tuning_large.c` - Large-scale threshold tuning
- `test_threshold.c` - Validate threshold behavior
