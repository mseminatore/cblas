//------------------------------------------------------
// Copyright 2023 Mark Seminatore. All rights reserved.
//
//------------------------------------------------------

#ifndef __CBLAS_H
#define __CBLAS_H

#include <stddef.h>
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

//[]------------------------------------------------------[]
// configurable library parameters
//[]------------------------------------------------------[]

// set the most threads supported by the library
#ifndef MAX_THREADS
#   define MAX_THREADS 64
#endif

// set the sizes of various buffers
#define CBLAS_SMALL_BUF 256
#define CBLAS_LARGE_BUF 1024

// library functions check inputs for validity
#define CBLAS_CHECK_INPUTS

// library functions use blas standard error reporting function
#define CBLAS_XERBLA_INPUTS

// uncomment to enable SSE SIMD instruction usage
#define USE_SSE

#define USE_SIMD

// uncomment to enable FMA3 instruction usage
//#define USE_INTEL_FMA

// uncomment to enable use of static vs. stack-based buffers
#define USE_STATIC_BUFFERS

// uncomment to enable multi-threading
#define MT_ENABLED

// uncomment to enable multi-threading debug messages
//#define MT_DEBUG

//------------------------------------------------------
// size type for indices
//------------------------------------------------------
typedef size_t CBLAS_INDEX;

#ifdef MT_DEBUG
#   define MT_TRACE(...) fprintf(stderr, __VA_ARGS__)
#else
#   define MT_TRACE(...)
// #   define MT_TRACE __noop
#endif

#define CBLAS_LEVEL_1_THREADING
#define CBLAS_LEVEL_2_THREADING
#define CBLAS_LEVEL_3_THREADING

#ifndef XERBLA
#   define XERBLA(param) xerbla(__func__, (param), strlen(__func__))
#endif

#ifndef TRUE
#   define TRUE 1
#endif

#ifndef FALSE
#   define FALSE 0
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
} work_queue_t;

//------------------------------------------------------
// BLAS Level 1 functions
//------------------------------------------------------
float cblas_sdot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);
double cblas_ddot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

void cblas_scopy(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);
void cblas_dcopy(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

void cblas_sscal(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx);
void cblas_dscal(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx);

void cblas_saxpy(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);
void cblas_daxpy(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

void cblas_sswap(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);
void cblas_dswap(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

void cblas_srot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float c, float s);
void cblas_drot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double c, double s);

float cblas_sasum(CBLAS_INDEX n, float *x, CBLAS_INDEX incx);
double cblas_dasum(CBLAS_INDEX n, double *x, CBLAS_INDEX incx);

float cblas_snrm2(CBLAS_INDEX n, float *x, CBLAS_INDEX incx);
double cblas_dnrm2(CBLAS_INDEX n, double *x, CBLAS_INDEX incx);

void cblas_srotg(float *a, float *b, float *c, float *s);
void cblas_drotg(double *a, double *b, double *c, double *s);

// non-standard extensions
void cblas_ssetv(CBLAS_INDEX n, float *x, float v);
void cblas_dsetv(CBLAS_INDEX n, double *x, double v);

void cblas_saxpby(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float beta, float *y, CBLAS_INDEX incy);
void cblas_daxpby(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double beta, double *y, CBLAS_INDEX incy);

//------------------------------------------------------
// BLAS Level 2 functions
//------------------------------------------------------
void cblas_sger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *a, CBLAS_INDEX lda);
void cblas_dger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double *a, CBLAS_INDEX lda);

void cblas_sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float *a, CBLAS_INDEX lda, float *x, CBLAS_INDEX incx, float beta, float *y, CBLAS_INDEX incy);
void cblas_dgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double *a, CBLAS_INDEX lda, double *x, CBLAS_INDEX incx, double beta, double *y, CBLAS_INDEX incy);

//------------------------------------------------------
// BLAS Level 3 functions
//------------------------------------------------------
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float beta, float *c, CBLAS_INDEX ldc);
void cblas_sgemm_naive(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float beta, float *c, CBLAS_INDEX ldc);
void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, double alpha, double *a, CBLAS_INDEX lda, double *b, CBLAS_INDEX ldb, double beta, double *c, CBLAS_INDEX ldc);

//------------------------------------------------------
// Utility functions
//------------------------------------------------------
void cblas_init(int threads);
void cblas_shutdown();
void cblas_set_num_threads(int threads);
int cblas_get_num_threads(void);
int cblas_init_server();

void cblas_execute(int items, work_queue_t* queue);
void cblas_execute_async(int items, work_queue_t* queue);
void cblas_execute_async_join(int items, work_queue_t* queue);

void cblas_level1_exec(CBLAS_INDEX stride, kernel_function kernel, CBLAS_INDEX n, void* x, CBLAS_INDEX incx, void* y, CBLAS_INDEX incy);

const char *cblas_get_config();
const char *cblas_get_corename();
int  cblas_get_num_procs();
void xerbla(const char *srcname, int info, size_t len);

//------------------------------------------------------
// CPU ID functions
//------------------------------------------------------
int cpu_get_core_count();
const char *cpu_get_core_name();
int cpu_get_cacheline_size();
unsigned int cpu_get_features();
const char *cblas_get_isa_features();
void cblas_print_configuration();

//------------------------------------------------------
// internal functions
//------------------------------------------------------

//------------------------------------------------------
// testing functions/structs
//------------------------------------------------------
struct cblas_timer
{
#ifdef _WIN32
    LARGE_INTEGER t;
#else
    struct timespec t;
#endif
};

int cblas_is_server_alive();
void cblas_set_server_alive(int yesno);
void cblas_timer_get_time(struct cblas_timer* t);
float cblas_timer_get_delta(struct cblas_timer* t1, struct cblas_timer* t2);

#ifdef __cplusplus
    }
#endif

#endif
