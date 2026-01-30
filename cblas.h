//------------------------------------------------------
// Cblas.h
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//
//------------------------------------------------------

#ifndef __CBLAS_H
#define __CBLAS_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#   include <Windows.h>
#endif

#ifdef __cplusplus
    extern "C" {
#endif

#ifndef LIB_CBLAS
#   define LIB_CBLAS
#endif

#define CBLAS_VERSION "0.16"

//[]------------------------------------------------------[]
// configurable library parameters
//[]------------------------------------------------------[]

// Include CMake-generated configuration
#include "cblas_config.h"

// set the sizes of various buffers
#define CBLAS_SMALL_BUF 256
#define CBLAS_LARGE_BUF 1024

// cache-aware block sizes for Level-2 operations
#define GEMV_BLOCK_SIZE 256  // Fits in L1 cache
#define GER_BLOCK_SIZE 256   // Fits in L1 cache

// uncomment to enable FMA3 instruction usage
//#define USE_INTEL_FMA

// uncomment to enable multi-threading debug messages
//#define MT_DEBUG

// uncomment to enable JSON format for MT debug output
//#define MT_DEBUG_JSON

// uncomment to enable performance counter tracking
#define CBLAS_ENABLE_STATS

// multi-threading threshold limits
#define CBLAS_MT_DOT    100000
#define CBLAS_MT_COPY   10000
#define CBLAS_MT_GER    10000
#define CBLAS_MT_GEMM   10000
#define CBLAS_MT_GEMV   10000

//------------------------------------------------------
// Compiler-independent macros for code quality
//------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
#   define CBLAS_UNUSED __attribute__((unused))
#   define CBLAS_FALLTHROUGH __attribute__((fallthrough))
#elif defined(_MSC_VER)
#   define CBLAS_UNUSED __pragma(warning(suppress: 4100 4101))
#   define CBLAS_FALLTHROUGH
#else
#   define CBLAS_UNUSED
#   define CBLAS_FALLTHROUGH
#endif

//------------------------------------------------------
// size type for indices
//------------------------------------------------------
typedef size_t CBLAS_INDEX;

#ifdef MT_DEBUG
#   ifdef MT_DEBUG_JSON
        // JSON output mode
#       define MT_TRACE(...) do { \
            fprintf(stderr, "{\"type\":\"trace\",\"message\":\""); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\"}\n"); \
        } while(0)
#       define MT_TRACE_THREAD(tid, ...) do { \
            fprintf(stderr, "{\"type\":\"thread\",\"tid\":%d,\"message\":\"", tid); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\"}\n"); \
        } while(0)
#       define MT_TRACE_TIMING(tid, op, duration_us) \
            fprintf(stderr, "{\"type\":\"timing\",\"tid\":%d,\"operation\":\"%s\",\"duration_us\":%.2f}\n", tid, op, duration_us)
#       define MT_TRACE_QUEUE_DEPTH(depth) \
            fprintf(stderr, "{\"type\":\"queue\",\"depth\":%d}\n", depth)
#   else
        // Human-readable output mode
#       define MT_TRACE(...) fprintf(stderr, __VA_ARGS__)
#       define MT_TRACE_THREAD(tid, ...) fprintf(stderr, "[Thread %d] ", tid); fprintf(stderr, __VA_ARGS__)
#       define MT_TRACE_TIMING(tid, op, duration_us) fprintf(stderr, "[Thread %d] %s took %.2f us\n", tid, op, duration_us)
#       define MT_TRACE_QUEUE_DEPTH(depth) fprintf(stderr, "[Queue] Depth: %d\n", depth)
#   endif
#   define MT_TRACE_LOAD_BALANCE(thread_count, times) do { \
        if ((thread_count) > 0) { \
            double min_time = times[0], max_time = times[0], sum = times[0]; \
            for (int _mt_idx = 1; _mt_idx < (thread_count); _mt_idx++) { \
                if (times[_mt_idx] < min_time) min_time = times[_mt_idx]; \
                if (times[_mt_idx] > max_time) max_time = times[_mt_idx]; \
                sum += times[_mt_idx]; \
            } \
            double avg_time = sum / (thread_count); \
            if (avg_time > 0.0) { \
                double variance = ((max_time - min_time) / avg_time) * 100.0; \
                int warning = (variance > 20.0) ? 1 : 0; \
                CBLAS_MT_TRACE_LOAD_BALANCE_IMPL(thread_count, variance, min_time, max_time, avg_time, warning); \
            } \
        } \
    } while(0)

#   ifdef MT_DEBUG_JSON
#       define CBLAS_MT_TRACE_LOAD_BALANCE_IMPL(count, variance, min_t, max_t, avg_t, warn) \
            fprintf(stderr, "{\"type\":\"load_balance\",\"thread_count\":%d,\"variance_pct\":%.1f,\"min_us\":%.2f,\"max_us\":%.2f,\"avg_us\":%.2f,\"warning\":%s}\n", \
                    count, variance, min_t, max_t, avg_t, warn ? "true" : "false")
#   else
#       define CBLAS_MT_TRACE_LOAD_BALANCE_IMPL(count, variance, min_t, max_t, avg_t, warn) \
            fprintf(stderr, "[Load Balance] %s: %.1f%% variance (min=%.2fus, max=%.2fus, avg=%.2fus)\n", \
                    warn ? "WARNING" : "OK", variance, min_t, max_t, avg_t)
#   endif

// Helper function to get current time in microseconds for MT debug timing
static inline double mt_get_time_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000000.0 + (double)ts.tv_nsec / 1000.0;
#endif
}
#else
#   define MT_TRACE(...)
#   define MT_TRACE_THREAD(tid, ...)
#   define MT_TRACE_TIMING(tid, op, duration_us)
#   define MT_TRACE_QUEUE_DEPTH(depth)
#   define MT_TRACE_LOAD_BALANCE(thread_count, times)
// #   define MT_TRACE __noop
#endif

// Performance counter macros
#ifdef CBLAS_ENABLE_STATS
#   define CBLAS_STATS_RECORD(op, n, mt) cblas_record_operation(op, n, mt, 0.0)
#   define CBLAS_STATS_START() struct cblas_timer _stats_t1, _stats_t2; cbu_timer_get_time(&_stats_t1)
#   define CBLAS_STATS_END(op, n, mt) do { \
        cbu_timer_get_time(&_stats_t2); \
        cblas_record_operation(op, n, mt, cbu_timer_get_delta(&_stats_t1, &_stats_t2)); \
    } while(0)
#else
#   define CBLAS_STATS_RECORD(op, n, mt)
#   define CBLAS_STATS_START()
#   define CBLAS_STATS_END(op, n, mt)
#endif

#define CBLAS_LEVEL_1_THREADING
#define CBLAS_LEVEL_2_THREADING
#define CBLAS_LEVEL_3_THREADING

#ifndef XERBLA
#   define XERBLA(param) xerbla(__func__, (param), strlen(__func__))
#endif

#ifndef CBLAS_TRUE
#   define CBLAS_TRUE 1
#endif

#ifndef CBLAS_FALSE
#   define CBLAS_FALSE 0
#endif

//------------------------------------------------------
// Input validation macros - reduce code duplication
//------------------------------------------------------

// Vector operations with single vector (n, x, incx)
#ifdef CBLAS_CHECK_INPUTS
#ifdef CBLAS_XERBLA_INPUTS
#define CBLAS_VALIDATE_VEC1(n, x, incx, ret) \
    do { \
        int info = 0; \
        if ((n) <= 0) \
            info = 1; \
        else if (!(x)) \
            info = 2; \
        else if ((incx) <= 0) \
            info = 3; \
        if (info) { \
            XERBLA(info); \
            return ret; \
        } \
    } while(0)
#else
#define CBLAS_VALIDATE_VEC1(n, x, incx, ret) \
    do { \
        if ((n) <= 0 || !(x) || (incx) <= 0) { \
            assert((n) > 0 && (x) && (incx) > 0); \
            return ret; \
        } \
    } while(0)
#endif
#else
#define CBLAS_VALIDATE_VEC1(n, x, incx, ret)
#endif

// Vector operations with two vectors (n, x, incx, y, incy)
#ifdef CBLAS_CHECK_INPUTS
#ifdef CBLAS_XERBLA_INPUTS
#define CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, ret) \
    do { \
        int info = 0; \
        if ((n) <= 0) \
            info = 1; \
        else if (!(x)) \
            info = 2; \
        else if (!(y)) \
            info = 4; \
        if (info) { \
            XERBLA(info); \
            return ret; \
        } \
    } while(0)
#else
#define CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, ret) \
    do { \
        if ((n) <= 0 || !(x) || !(y)) { \
            assert((n) > 0 && (x) && (y)); \
            return ret; \
        } \
    } while(0)
#endif
#else
#define CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, ret)
#endif

// Vector scaling with alpha parameter (n, alpha, x, incx)
#ifdef CBLAS_CHECK_INPUTS
#ifdef CBLAS_XERBLA_INPUTS
#define CBLAS_VALIDATE_SCAL(n, alpha, x, incx, ret) \
    do { \
        int info = 0; \
        if ((n) <= 0) \
            info = 1; \
        else if (!(x)) \
            info = 3; \
        else if ((incx) <= 0) \
            info = 4; \
        if (info) { \
            XERBLA(info); \
            return ret; \
        } \
    } while(0)
#else
#define CBLAS_VALIDATE_SCAL(n, alpha, x, incx, ret) \
    do { \
        if ((n) <= 0 || !(x) || (incx) <= 0) { \
            assert((n) > 0 && (x) && (incx) > 0); \
            return ret; \
        } \
    } while(0)
#endif
#else
#define CBLAS_VALIDATE_SCAL(n, alpha, x, incx, ret)
#endif

// AXPY operations (n, alpha, x, incx, y, incy)
#ifdef CBLAS_CHECK_INPUTS
#ifdef CBLAS_XERBLA_INPUTS
#define CBLAS_VALIDATE_AXPY(n, alpha, x, incx, y, incy, ret) \
    do { \
        int info = 0; \
        if ((n) <= 0) \
            info = 1; \
        else if ((alpha) == 0.0) \
            info = 2; \
        else if (!(x)) \
            info = 3; \
        else if (!(y)) \
            info = 5; \
        if (info) { \
            XERBLA(info); \
            return ret; \
        } \
    } while(0)
#else
#define CBLAS_VALIDATE_AXPY(n, alpha, x, incx, y, incy, ret) \
    do { \
        if ((n) <= 0 || (alpha) == 0.0 || !(x) || !(y)) { \
            assert((n) > 0 && (alpha) != 0.0 && (x) && (y)); \
            return ret; \
        } \
    } while(0)
#endif
#else
#define CBLAS_VALIDATE_AXPY(n, alpha, x, incx, y, incy, ret)
#endif

// AXPBY operations (n, alpha, x, incx, beta, y, incy)
#ifdef CBLAS_CHECK_INPUTS
#ifdef CBLAS_XERBLA_INPUTS
#define CBLAS_VALIDATE_AXPBY(n, alpha, x, incx, beta, y, incy, ret) \
    do { \
        int info = 0; \
        if ((n) <= 0) \
            info = 1; \
        else if ((alpha) == 0.0) \
            info = 2; \
        else if (!(x)) \
            info = 3; \
        else if ((beta) == 0.0) \
            info = 5; \
        else if (!(y)) \
            info = 6; \
        if (info) { \
            XERBLA(info); \
            return ret; \
        } \
    } while(0)
#else
#define CBLAS_VALIDATE_AXPBY(n, alpha, x, incx, beta, y, incy, ret) \
    do { \
        if ((n) <= 0 || (alpha) == 0.0 || !(x) || (beta) == 0.0 || !(y)) { \
            assert((n) > 0 && (alpha) != 0.0 && (x) && (beta) != 0.0 && (y)); \
            return ret; \
        } \
    } while(0)
#endif
#else
#define CBLAS_VALIDATE_AXPBY(n, alpha, x, incx, beta, y, incy, ret)
#endif

// GER matrix-vector operations
#ifdef CBLAS_CHECK_INPUTS
#ifdef CBLAS_XERBLA_INPUTS
#define CBLAS_VALIDATE_GER(layout, m, n, x, incx, y, incy, a, lda, ret) \
    do { \
        int info = 0; \
        if ((layout) != CblasRowMajor && (layout) != CblasColMajor) \
            info = 1; \
        else if (!(x)) \
            info = 5; \
        else if ((incx) == 0) \
            info = 6; \
        else if (!(y)) \
            info = 7; \
        else if ((incy) == 0) \
            info = 8; \
        else if (!(a)) \
            info = 9; \
        else if ((lda) < MAX(1, (m))) \
            info = 10; \
        if (info) { \
            XERBLA(info); \
            return ret; \
        } \
    } while(0)
#else
#define CBLAS_VALIDATE_GER(layout, m, n, x, incx, y, incy, a, lda, ret) \
    do { \
        if ((m) < 0 || (n) < 0 || !(x) || (incx) == 0 || (incy) == 0 || !(a) || (lda) < MAX(1, (m))) { \
            assert((m) > 0 && (n) > 0 && (incx) != 0 && (incy) != 0); \
            assert((x) && (y) && (a)); \
            assert((layout) == CblasRowMajor || (layout) == CblasColMajor); \
            assert((lda) >= MAX(1, (m))); \
            return ret; \
        } \
    } while(0)
#endif
#else
#define CBLAS_VALIDATE_GER(layout, m, n, x, incx, y, incy, a, lda, ret)
#endif

#ifndef MAX
#   define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#   define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef CLAMP
#   define CLAMP(value, min, max)  ((value) < (min) ? (min) : ((value) > (max) ? (max) : (value)))
#endif

#ifndef CHECK_ALIGN
    #ifdef NDEBUG
    #   define CHECK_ALIGN(p, align)
    #else
    #   define CHECK_ALIGN(p, align) assert(!(((intptr_t)p) & ((align) - 1)))
    #endif
#endif

//#if !defined(__STDC_NO_ATOMICS__)
//#   include <stdatomic.h>
//#   define MB atomic_thread_fence(memory_order_relaxed)
//#else
//#   error C11 is required!
//#   define MB
//#endif

//------------------------------------------------------
// enumeration for matrix layouts
//------------------------------------------------------
typedef enum CBLAS_LAYOUT {
    CblasRowMajor,
    CblasColMajor
} CBLAS_LAYOUT;

//------------------------------------------------------
// enumeration for transposing matrices
//------------------------------------------------------
typedef enum CBLAS_TRANSPOSE {
    CblasTrans,
    CblasNoTrans
} CBLAS_TRANSPOSE;

//------------------------------------------------------
// mode indicator for BLAS level
//------------------------------------------------------
enum {
    CblasLevel1,
    CblasLevel2,
    CblasLevel3
};

//------------------------------------------------------
// CPU features
//------------------------------------------------------
#define CPU_NONE        0x00
#define CPU_SSE         0x01
#define CPU_AVX         0x02
#define CPU_AVX2        0x04
#define CPU_AVX512      0x08
#define CPU_x64_FMA3    0x10
#define CPU_NEON        0x20
#define CPU_NEON_FMA    0x40

//------------------------------------------------------
// arguments passed to kernel functions
//------------------------------------------------------
typedef struct
{
    CBLAS_INDEX m, n, k, incx, incy, lda, ldb, ldc, ib, pb;
    void *x, *y, *c, *alpha, *beta, *a, *b;
} cblas_args_t;

// kernel operation for MT tasks
typedef void (*kernel_function)(cblas_args_t* args);

typedef struct
{
    kernel_function sgemm_k;
    kernel_function dgemm_k;
    kernel_function sger_k;
    kernel_function dger_k;
    kernel_function sgemv_k;
    kernel_function dgemv_k;
} kernels_t;

#define CBLAS_DEFAULT_THREADS -1

//------------------------------------------------------
// structure defining the work queue format
//------------------------------------------------------
typedef struct work_queue_t
{
    struct work_queue_t* next;  // ptr to next task in queue

    cblas_args_t* args;         // parameters to kernel function
    int type;                   // type of call

#if 1
    volatile int finished;      // this work item is finished
#else
    atomic_int finished;
#endif

    int thread_num, tid;

    kernel_function kernel;

#ifdef MT_DEBUG
    // Timing information for debug/profiling
    double start_time_us;       // when task started (microseconds)
    double end_time_us;         // when task completed (microseconds)
    const char* operation;      // operation name for debugging (e.g., "DOT", "COPY", "GEMM")
#endif
} work_queue_t;

//------------------------------------------------------
// BLAS Level 1 functions
//------------------------------------------------------

/**
 * @brief Compute dot product of two single-precision vectors
 * @param n Number of elements (must be > 0)
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Input vector Y (must be non-NULL)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @return Dot product result (X · Y)
 * @note Thread-safe. Uses MT when n > CBLAS_MT_DOT.
 */
float cblas_sdot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);

/**
 * @brief Compute dot product of two double-precision vectors
 * @param n Number of elements (must be > 0)
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Input vector Y (must be non-NULL)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @return Dot product result (X · Y)
 * @note Thread-safe. Uses MT when n > CBLAS_MT_DOT.
 */
double cblas_ddot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

/**
 * @brief Copy single-precision vector X to vector Y
 * @param n Number of elements (must be > 0)
 * @param x Source vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Destination vector Y (must be non-NULL)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @note Thread-safe. Uses MT when n > CBLAS_MT_COPY. Y = X
 */
void cblas_scopy(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);

/**
 * @brief Copy double-precision vector X to vector Y
 * @param n Number of elements (must be > 0)
 * @param x Source vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Destination vector Y (must be non-NULL)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @note Thread-safe. Uses MT when n > CBLAS_MT_COPY. Y = X
 */
void cblas_dcopy(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

/**
 * @brief Scale single-precision vector X by scalar alpha
 * @param n Number of elements (must be > 0)
 * @param alpha Scalar multiplier
 * @param x Vector X to be scaled (must be non-NULL, modified in place)
 * @param incx Stride for X (typically 1 for contiguous)
 * @note Thread-safe. X = alpha * X
 */
void cblas_sscal(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx);

/**
 * @brief Scale double-precision vector X by scalar alpha
 * @param n Number of elements (must be > 0)
 * @param alpha Scalar multiplier
 * @param x Vector X to be scaled (must be non-NULL, modified in place)
 * @param incx Stride for X (typically 1 for contiguous)
 * @note Thread-safe. X = alpha * X
 */
void cblas_dscal(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx);

/**
 * @brief Single-precision constant times a vector plus a vector
 * @param n Number of elements (must be > 0)
 * @param alpha Scalar multiplier
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Vector Y (must be non-NULL, modified in place)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @note Thread-safe. Y = alpha * X + Y
 */
void cblas_saxpy(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);

/**
 * @brief Double-precision constant times a vector plus a vector
 * @param n Number of elements (must be > 0)
 * @param alpha Scalar multiplier
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Vector Y (must be non-NULL, modified in place)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @note Thread-safe. Y = alpha * X + Y
 */
void cblas_daxpy(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

/**
 * @brief Swap single-precision vectors X and Y
 * @param n Number of elements (must be > 0)
 * @param x Vector X (must be non-NULL, modified in place)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Vector Y (must be non-NULL, modified in place)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @note Thread-safe. X <-> Y
 */
void cblas_sswap(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);

/**
 * @brief Swap double-precision vectors X and Y
 * @param n Number of elements (must be > 0)
 * @param x Vector X (must be non-NULL, modified in place)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Vector Y (must be non-NULL, modified in place)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @note Thread-safe. X <-> Y
 */
void cblas_dswap(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

/**
 * @brief Apply Givens rotation to single-precision vectors
 * @param n Number of elements (must be > 0)
 * @param x Vector X (must be non-NULL, modified in place)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Vector Y (must be non-NULL, modified in place)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @param c Cosine of rotation angle
 * @param s Sine of rotation angle
 * @note Thread-safe. Applies plane rotation: x' = c*x + s*y, y' = c*y - s*x
 */
void cblas_srot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float c, float s);

/**
 * @brief Apply Givens rotation to double-precision vectors
 * @param n Number of elements (must be > 0)
 * @param x Vector X (must be non-NULL, modified in place)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Vector Y (must be non-NULL, modified in place)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @param c Cosine of rotation angle
 * @param s Sine of rotation angle
 * @note Thread-safe. Applies plane rotation: x' = c*x + s*y, y' = c*y - s*x
 */
void cblas_drot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double c, double s);

/**
 * @brief Sum of absolute values of single-precision vector elements
 * @param n Number of elements (must be > 0)
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @return Sum of |x[i]| for all elements
 * @note Thread-safe.
 */
float cblas_sasum(CBLAS_INDEX n, float *x, CBLAS_INDEX incx);

/**
 * @brief Sum of absolute values of double-precision vector elements
 * @param n Number of elements (must be > 0)
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @return Sum of |x[i]| for all elements
 * @note Thread-safe.
 */
double cblas_dasum(CBLAS_INDEX n, double *x, CBLAS_INDEX incx);

/**
 * @brief Euclidean norm (L2 norm) of single-precision vector
 * @param n Number of elements (must be > 0)
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @return Euclidean norm: sqrt(sum of x[i]^2)
 * @note Thread-safe.
 */
float cblas_snrm2(CBLAS_INDEX n, float *x, CBLAS_INDEX incx);

/**
 * @brief Euclidean norm (L2 norm) of double-precision vector
 * @param n Number of elements (must be > 0)
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @return Euclidean norm: sqrt(sum of x[i]^2)
 * @note Thread-safe.
 */
double cblas_dnrm2(CBLAS_INDEX n, double *x, CBLAS_INDEX incx);

/**
 * @brief Construct Givens rotation for single-precision values
 * @param a Pointer to first value (modified to contain r on output)
 * @param b Pointer to second value (modified to contain z on output)
 * @param c Pointer to store cosine of rotation angle
 * @param s Pointer to store sine of rotation angle
 * @note Computes rotation such that [c s; -s c] * [a; b] = [r; 0]
 */
void cblas_srotg(float *a, float *b, float *c, float *s);

/**
 * @brief Construct Givens rotation for double-precision values
 * @param a Pointer to first value (modified to contain r on output)
 * @param b Pointer to second value (modified to contain z on output)
 * @param c Pointer to store cosine of rotation angle
 * @param s Pointer to store sine of rotation angle
 * @note Computes rotation such that [c s; -s c] * [a; b] = [r; 0]
 */
void cblas_drotg(double *a, double *b, double *c, double *s);

// non-standard extensions
/**
 * @brief Set all elements of single-precision vector to a value
 * @param n Number of elements (must be > 0)
 * @param x Vector X (must be non-NULL, modified in place)
 * @param v Value to set
 * @note Non-standard extension. Thread-safe. X[i] = v for all i
 */
void cblas_ssetv(CBLAS_INDEX n, float *x, float v);

/**
 * @brief Set all elements of double-precision vector to a value
 * @param n Number of elements (must be > 0)
 * @param x Vector X (must be non-NULL, modified in place)
 * @param v Value to set
 * @note Non-standard extension. Thread-safe. X[i] = v for all i
 */
void cblas_dsetv(CBLAS_INDEX n, double *x, double v);

/**
 * @brief Single-precision alpha*X plus beta*Y
 * @param n Number of elements (must be > 0)
 * @param alpha Scalar multiplier for X
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param beta Scalar multiplier for Y
 * @param y Vector Y (must be non-NULL, modified in place)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @note Non-standard extension. Thread-safe. Y = alpha * X + beta * Y
 */
void cblas_saxpby(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float beta, float *y, CBLAS_INDEX incy);

/**
 * @brief Double-precision alpha*X plus beta*Y
 * @param n Number of elements (must be > 0)
 * @param alpha Scalar multiplier for X
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param beta Scalar multiplier for Y
 * @param y Vector Y (must be non-NULL, modified in place)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @note Non-standard extension. Thread-safe. Y = alpha * X + beta * Y
 */
void cblas_daxpby(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double beta, double *y, CBLAS_INDEX incy);

//------------------------------------------------------
// BLAS Level 2 functions
//------------------------------------------------------

/**
 * @brief Single-precision general matrix rank-1 update
 * @param layout Matrix storage layout (CblasRowMajor or CblasColMajor)
 * @param m Number of rows in matrix A (must be > 0)
 * @param n Number of columns in matrix A (must be > 0)
 * @param alpha Scalar multiplier
 * @param x Input vector X of length m (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Input vector Y of length n (must be non-NULL)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @param a Matrix A (must be non-NULL, modified in place)
 * @param lda Leading dimension of A (must be >= n for row-major, >= m for col-major)
 * @note Thread-safe. Uses MT when m*n > CBLAS_MT_GER. A = alpha * X * Y^T + A
 */
void cblas_sger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *a, CBLAS_INDEX lda);

/**
 * @brief Double-precision general matrix rank-1 update
 * @param layout Matrix storage layout (CblasRowMajor or CblasColMajor)
 * @param m Number of rows in matrix A (must be > 0)
 * @param n Number of columns in matrix A (must be > 0)
 * @param alpha Scalar multiplier
 * @param x Input vector X of length m (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param y Input vector Y of length n (must be non-NULL)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @param a Matrix A (must be non-NULL, modified in place)
 * @param lda Leading dimension of A (must be >= n for row-major, >= m for col-major)
 * @note Thread-safe. Uses MT when m*n > CBLAS_MT_GER. A = alpha * X * Y^T + A
 */
void cblas_dger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double *a, CBLAS_INDEX lda);

/**
 * @brief Single-precision general matrix-vector multiplication
 * @param layout Matrix storage layout (CblasRowMajor or CblasColMajor)
 * @param trans Operation on matrix A (CblasTrans or CblasNoTrans)
 * @param m Number of rows in matrix A (must be > 0)
 * @param n Number of columns in matrix A (must be > 0)
 * @param alpha Scalar multiplier for A*X
 * @param a Matrix A (must be non-NULL)
 * @param lda Leading dimension of A (must be >= n for row-major, >= m for col-major)
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param beta Scalar multiplier for Y
 * @param y Vector Y (must be non-NULL, modified in place)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @note Thread-safe. Uses MT when m*n > CBLAS_MT_GEMV. Y = alpha * op(A) * X + beta * Y
 */
void cblas_sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float *a, CBLAS_INDEX lda, float *x, CBLAS_INDEX incx, float beta, float *y, CBLAS_INDEX incy);

/**
 * @brief Double-precision general matrix-vector multiplication
 * @param layout Matrix storage layout (CblasRowMajor or CblasColMajor)
 * @param trans Operation on matrix A (CblasTrans or CblasNoTrans)
 * @param m Number of rows in matrix A (must be > 0)
 * @param n Number of columns in matrix A (must be > 0)
 * @param alpha Scalar multiplier for A*X
 * @param a Matrix A (must be non-NULL)
 * @param lda Leading dimension of A (must be >= n for row-major, >= m for col-major)
 * @param x Input vector X (must be non-NULL)
 * @param incx Stride for X (typically 1 for contiguous)
 * @param beta Scalar multiplier for Y
 * @param y Vector Y (must be non-NULL, modified in place)
 * @param incy Stride for Y (typically 1 for contiguous)
 * @note Thread-safe. Uses MT when m*n > CBLAS_MT_GEMV. Y = alpha * op(A) * X + beta * Y
 */
void cblas_dgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double *a, CBLAS_INDEX lda, double *x, CBLAS_INDEX incx, double beta, double *y, CBLAS_INDEX incy);

//------------------------------------------------------
// BLAS Level 3 functions
//------------------------------------------------------

/**
 * @brief Single-precision general matrix-matrix multiplication
 * @param layout Matrix storage layout (CblasRowMajor or CblasColMajor)
 * @param transa Operation on matrix A (CblasTrans or CblasNoTrans)
 * @param transb Operation on matrix B (CblasTrans or CblasNoTrans)
 * @param m Number of rows in op(A) and C (must be > 0)
 * @param n Number of columns in op(B) and C (must be > 0)
 * @param k Number of columns in op(A) and rows in op(B) (must be > 0)
 * @param alpha Scalar multiplier for A*B
 * @param a Matrix A (must be non-NULL)
 * @param lda Leading dimension of A
 * @param b Matrix B (must be non-NULL)
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param c Matrix C (must be non-NULL, modified in place)
 * @param ldc Leading dimension of C
 * @note Thread-safe. Uses MT when m*n*k > CBLAS_MT_GEMM. C = alpha * op(A) * op(B) + beta * C
 * @note Optimized with cache blocking and SIMD. Uses FMA if available.
 */
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float beta, float *c, CBLAS_INDEX ldc);

/**
 * @brief Single-precision general matrix-matrix multiplication (naive reference)
 * @param layout Matrix storage layout (CblasRowMajor or CblasColMajor)
 * @param transa Operation on matrix A (CblasTrans or CblasNoTrans)
 * @param transb Operation on matrix B (CblasTrans or CblasNoTrans)
 * @param m Number of rows in op(A) and C (must be > 0)
 * @param n Number of columns in op(B) and C (must be > 0)
 * @param k Number of columns in op(A) and rows in op(B) (must be > 0)
 * @param alpha Scalar multiplier for A*B
 * @param a Matrix A (must be non-NULL)
 * @param lda Leading dimension of A
 * @param b Matrix B (must be non-NULL)
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param c Matrix C (must be non-NULL, modified in place)
 * @param ldc Leading dimension of C
 * @note Unoptimized reference implementation. Use cblas_sgemm for production.
 * @note C = alpha * op(A) * op(B) + beta * C
 */
void cblas_sgemm_naive(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float beta, float *c, CBLAS_INDEX ldc);

/**
 * @brief Double-precision general matrix-matrix multiplication
 * @param layout Matrix storage layout (CblasRowMajor or CblasColMajor)
 * @param transa Operation on matrix A (CblasTrans or CblasNoTrans)
 * @param transb Operation on matrix B (CblasTrans or CblasNoTrans)
 * @param m Number of rows in op(A) and C (must be > 0)
 * @param n Number of columns in op(B) and C (must be > 0)
 * @param k Number of columns in op(A) and rows in op(B) (must be > 0)
 * @param alpha Scalar multiplier for A*B
 * @param a Matrix A (must be non-NULL)
 * @param lda Leading dimension of A
 * @param b Matrix B (must be non-NULL)
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param c Matrix C (must be non-NULL, modified in place)
 * @param ldc Leading dimension of C
 * @note Thread-safe. Uses MT when m*n*k > CBLAS_MT_GEMM. C = alpha * op(A) * op(B) + beta * C
 * @note Optimized with cache blocking and SIMD.
 */
void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, double alpha, double *a, CBLAS_INDEX lda, double *b, CBLAS_INDEX ldb, double beta, double *c, CBLAS_INDEX ldc);

//------------------------------------------------------
// Utility functions
//------------------------------------------------------

/**
 * @brief Initialize CBLAS library and thread server
 * @param threads Number of threads to use (or CBLAS_DEFAULT_THREADS for auto-detect)
 * @note Must be called before using any BLAS functions. Thread-safe.
 * @note CBLAS_DEFAULT_THREADS (-1) uses the number of CPU cores
 */
void cblas_init(int threads);

/**
 * @brief Shutdown CBLAS library and cleanup thread server
 * @note Should be called when finished using BLAS functions to release resources.
 * @note Waits for all threads to complete and releases synchronization primitives.
 */
void cblas_shutdown(void);

/**
 * @brief Set the number of worker threads
 * @param threads Number of threads to use (1 to MAX_THREADS)
 * @note Thread-safe. Can be called at runtime to adjust parallelism.
 * @note Setting to 1 effectively disables multi-threading.
 */
void cblas_set_num_threads(int threads);

/**
 * @brief Get the current number of worker threads
 * @return Current thread count
 * @note Thread-safe.
 */
int cblas_get_num_threads(void);

/**
 * @brief Initialize thread server (internal use)
 * @return Status code (non-zero on success)
 * @note For internal use. Called by cblas_init().
 */
int cblas_init_server(void);

/**
 * @brief Execute work queue synchronously
 * @param items Number of work items in queue
 * @param queue Pointer to work queue
 * @note For internal use by BLAS functions.
 */
void cblas_execute(CBLAS_INDEX items, work_queue_t* queue);

/**
 * @brief Execute work queue asynchronously
 * @param items Number of work items in queue
 * @param queue Pointer to work queue
 * @note For internal use by BLAS functions.
 */
void cblas_execute_async(CBLAS_INDEX items, work_queue_t* queue);

/**
 * @brief Execute work queue asynchronously and wait for completion
 * @param items Number of work items in queue
 * @param queue Pointer to work queue
 * @note For internal use by BLAS functions.
 */
void cblas_execute_async_join(CBLAS_INDEX items, work_queue_t* queue);

/**
 * @brief Execute Level-1 BLAS operation (internal)
 * @param stride Byte stride for elements
 * @param kernel Kernel function to execute
 * @param n Number of elements
 * @param x Pointer to vector X
 * @param incx Stride for X
 * @param y Pointer to vector Y
 * @param incy Stride for Y
 * @param op_name Operation name for debug output (can be NULL)
 * @note For internal use by Level-1 BLAS functions.
 */
void cblas_level1_exec(CBLAS_INDEX stride, kernel_function kernel, CBLAS_INDEX n, void* x, CBLAS_INDEX incx, void* y, CBLAS_INDEX incy, const char* op_name);

/**
 * @brief Execute Level-1 BLAS operation with result (internal)
 * @param byte_stride Byte stride for elements
 * @param kernel Kernel function to execute
 * @param n Number of elements
 * @param x Pointer to vector X
 * @param incx Stride for X
 * @param y Pointer to vector Y
 * @param incy Stride for Y
 * @param c Pointer to result storage
 * @param op_name Operation name for debug output (can be NULL)
 * @note For internal use by Level-1 BLAS functions that return values.
 */
void cblas_level1_exec_result(CBLAS_INDEX byte_stride, kernel_function kernel, CBLAS_INDEX n, void* x, CBLAS_INDEX incx, void* y, CBLAS_INDEX incy, void* c, const char* op_name);

/**
 * @brief Get library configuration string
 * @return Configuration string describing compile-time options
 * @note Thread-safe.
 */
const char *cblas_get_config(void);

/**
 * @brief Get CPU core name
 * @return String describing CPU model
 * @note Thread-safe.
 */
const char *cblas_get_corename(void);

/**
 * @brief Get number of CPU cores
 * @return Number of logical processors
 * @note Thread-safe.
 */
int  cblas_get_num_procs(void);

/**
 * @brief BLAS standard error reporting function
 * @param srcname Name of function reporting error
 * @param info Parameter number that caused error
 * @param len Length of srcname string
 * @note For internal use by BLAS functions when CBLAS_XERBLA_INPUTS is defined.
 */
void xerbla(const char *srcname, int info, size_t len);

//------------------------------------------------------
// CPU ID functions
//------------------------------------------------------

/**
 * @brief Get number of CPU cores
 * @return Number of logical processors/cores
 * @note Thread-safe. Platform-specific implementation.
 */
int cpu_get_core_count(void);

/**
 * @brief Get CPU core name/model
 * @return String describing CPU model (e.g., "Intel Core i7")
 * @note Thread-safe. Returns brand string from CPUID.
 */
const char *cpu_get_core_name(void);

/**
 * @brief Get CPU cache line size
 * @return Cache line size in bytes (typically 64)
 * @note Thread-safe. Used for alignment optimization.
 */
int cpu_get_cacheline_size(void);

/**
 * @brief Get CPU feature flags
 * @return Bitmask of CPU features (CPU_SSE, CPU_AVX, CPU_AVX2, CPU_NEON, etc.)
 * @note Thread-safe. Used for runtime ISA detection and kernel dispatch.
 */
unsigned int cpu_get_features(void);

/**
 * @brief Get human-readable ISA features string
 * @return Comma-separated list of supported ISA extensions
 * @note Thread-safe. Returns string like "SSE, AVX2, FMA3".
 */
const char *cblas_get_isa_features(void);

/**
 * @brief Print library configuration to stdout
 * @note Displays version, threads, CPU info, ISA features.
 */
void cblas_print_configuration(void);

/**
 * @brief Get L2 cache size
 * @return L2 cache size in bytes
 * @note Thread-safe. Used for cache blocking tuning.
 */
int cpu_get_l2_cache_size(void);

//------------------------------------------------------
// internal functions
//------------------------------------------------------

/**
 * @brief Check if thread server is alive (internal)
 * @return Non-zero if server is running
 * @note For internal use only.
 */
int cblas_is_server_alive(void);

/**
 * @brief Set thread server alive status (internal)
 * @param yesno Non-zero to mark alive, zero to mark dead
 * @note For internal use only.
 */
void cblas_set_server_alive(int yesno);

/**
 * @brief Record a BLAS operation for performance tracking (internal)
 * @param operation Name of BLAS operation (e.g., "sdot", "sgemm")
 * @param elements Number of elements processed
 * @param mt_used Non-zero if multi-threading was used
 * @param time_sec Execution time in seconds
 * @note For internal use only. Used by BLAS functions to track statistics.
 */
void cblas_record_operation(const char* operation, uint64_t elements, int mt_used, double time_sec);

//------------------------------------------------------
// Performance counters
//------------------------------------------------------

/**
 * @brief Performance statistics for a BLAS operation
 * @note Tracks call count, total elements processed, MT activations, and total time.
 */
typedef struct {
    uint64_t total_calls;       /**< Total number of calls to this operation */
    uint64_t total_elements;    /**< Total number of elements processed */
    uint64_t mt_activations;    /**< Number of times multi-threading was activated */
    double total_time_sec;      /**< Total execution time in seconds */
} cblas_stats_t;

/**
 * @brief Get performance statistics for a specific operation
 * @param operation Name of BLAS operation (e.g., "sdot", "sgemm")
 * @return Pointer to statistics structure, or NULL if operation not found
 * @note Thread-safe. Returns pointer to internal static data.
 */
const cblas_stats_t* cblas_get_stats(const char* operation);

/**
 * @brief Reset all performance counters to zero
 * @note Thread-safe. Clears all accumulated statistics.
 */
void cblas_reset_stats(void);

/**
 * @brief Print performance statistics to stdout
 * @note Displays stats for all operations with non-zero call counts.
 */
void cblas_print_stats(void);

/**
 * @brief Cleanup stats resources
 * @note Called internally by cblas_shutdown(). Not for direct user call.
 */
void cblas_cleanup_stats(void);

//------------------------------------------------------
// testing functions/structs
//------------------------------------------------------

/**
 * @brief High-resolution timer structure
 * @note Platform-specific implementation (LARGE_INTEGER on Windows, timespec on POSIX)
 */
struct cblas_timer
{
#ifdef _WIN32
    LARGE_INTEGER t;
#else
    struct timespec t;
#endif
};

/**
 * @brief Get current time
 * @param t Pointer to timer structure to populate
 * @note For benchmarking and performance testing.
 */
void cbu_timer_get_time(struct cblas_timer* t);

/**
 * @brief Calculate elapsed time between two timestamps
 * @param t1 Start time
 * @param t2 End time
 * @return Elapsed time in seconds (floating-point)
 * @note For benchmarking and performance testing.
 */
float cbu_timer_get_delta(struct cblas_timer* t1, struct cblas_timer* t2);

/**
 * @brief Set single-precision matrix to identity
 * @param mtx Matrix pointer (must be non-NULL)
 * @param cols Number of columns
 * @param rows Number of rows
 * @note Utility function for testing. Sets diagonal to 1.0, off-diagonal to 0.0.
 */
void cbu_sge_set_identity(float* mtx, CBLAS_INDEX cols, CBLAS_INDEX rows);

/**
 * @brief Set double-precision matrix to identity
 * @param mtx Matrix pointer (must be non-NULL)
 * @param cols Number of columns
 * @param rows Number of rows
 * @note Utility function for testing. Sets diagonal to 1.0, off-diagonal to 0.0.
 */
void cbu_dge_set_identity(double* mtx, CBLAS_INDEX cols, CBLAS_INDEX rows);

/**
 * @brief Check if single-precision matrix is identity
 * @param mtx Matrix pointer (must be non-NULL)
 * @param cols Number of columns
 * @param rows Number of rows
 * @return Non-zero if matrix is identity, zero otherwise
 * @note Utility function for testing.
 */
int cbu_sge_is_identity(float* mtx, CBLAS_INDEX cols, CBLAS_INDEX rows);

/**
 * @brief Check if double-precision matrix is identity
 * @param mtx Matrix pointer (must be non-NULL)
 * @param cols Number of columns
 * @param rows Number of rows
 * @return Non-zero if matrix is identity, zero otherwise
 * @note Utility function for testing.
 */
int cbu_dge_is_identity(double* mtx, CBLAS_INDEX cols, CBLAS_INDEX rows);

/**
 * @brief Allocate and create single-precision identity matrix
 * @param cols Number of columns
 * @param rows Number of rows
 * @return Pointer to newly allocated identity matrix
 * @note Caller must free returned pointer. Utility function for testing.
 */
float *cbu_sge_make_identity(int cols, int rows);

/**
 * @brief Allocate and create double-precision identity matrix
 * @param cols Number of columns
 * @param rows Number of rows
 * @return Pointer to newly allocated identity matrix
 * @note Caller must free returned pointer. Utility function for testing.
 */
double* cbu_dge_make_identity(int cols, int rows);

//------------------------------------------------------
// kernel dispatch
//------------------------------------------------------

/**
 * @brief Global kernel function dispatch table
 * @note Populated at runtime based on CPU features by cpu_get_features().
 * @note Contains optimized kernel pointers for different ISA extensions.
 */
extern kernels_t blas_kernels;

/**
 * @brief Single-precision GEMM kernel (standard implementation)
 * @param args Kernel arguments structure
 * @note For internal use. Called via blas_kernels dispatch table.
 */
void sgemm_k(cblas_args_t* args);

#if defined(USE_SSE) && defined(USE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))
/**
 * @brief Single-precision GEMM kernel with FMA instructions
 * @param args Kernel arguments structure
 * @note For internal use. Used when CPU_x64_FMA3 feature is detected.
 * @note Requires FMA3 instruction set (Haswell or newer).
 */
void sgemm_k_fma(cblas_args_t* args);
#endif

/**
 * @brief Single-precision GER kernel (rank-1 update)
 * @param args Kernel arguments structure
 * @note For internal use. Called via blas_kernels dispatch table.
 */
void sger_k(cblas_args_t* args);

/**
 * @brief Double-precision GER kernel (rank-1 update)
 * @param args Kernel arguments structure
 * @note For internal use. Called via blas_kernels dispatch table.
 */
void dger_k(cblas_args_t* args);

/**
 * @brief Single-precision GEMV kernel (matrix-vector multiply)
 * @param args Kernel arguments structure
 * @note For internal use. Called via blas_kernels dispatch table.
 */
void sgemv_k(cblas_args_t* args);

/**
 * @brief Double-precision GEMV kernel (matrix-vector multiply)
 * @param args Kernel arguments structure
 * @note For internal use. Called via blas_kernels dispatch table.
 */
void dgemv_k(cblas_args_t* args);

#ifdef __cplusplus
    }
#endif

#endif
