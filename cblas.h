//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#ifndef __CBLAS_H
#define __CBLAS_H

#include <stddef.h>
#include <math.h>
#include <assert.h>

#ifndef MAX_THREADS
#   define MAX_THREADS 64
#endif

#ifdef _WIN32
#   define MT_ENABLED
#endif

//#define MT_DEBUG

#ifdef MT_DEBUG
#   define MT_TRACE printf
#else
#   define MT_TRACE __noop
#endif

//------------------------------------------------------
//
//------------------------------------------------------
typedef size_t CBLAS_INDEX;

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
// arguments passed to kernel functions
//------------------------------------------------------
typedef struct
{
    CBLAS_INDEX m, n, k, incx, incy, lda, ldb;
    void* x, * y, * alpha, * beta;
} cblas_args_t;

typedef void (*kernel_function)(cblas_args_t* args);

//------------------------------------------------------
// structure defining the work queue format
//------------------------------------------------------
typedef struct work_queue_t
{
    struct work_queue_t* next;  // ptr to next task in queue

    cblas_args_t* args;         // parameters to kernel function
    int type;                   // type of call
    volatile int finished;      // this work item is finished
    int thread_num, tid;

    kernel_function kernel;
} work_queue_t;

//------------------------------------------------------
// BLAS Level 1 functions
//------------------------------------------------------
float cblas_sdot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, const float *y, CBLAS_INDEX incy);
double cblas_ddot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, const double *y, CBLAS_INDEX incy);

void cblas_scopy(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, const float *y, CBLAS_INDEX incy);
void cblas_dcopy(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, const double *y, CBLAS_INDEX incy);

void cblas_sscal(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx);
void cblas_dscal(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx);

void cblas_saxpy(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);
void cblas_daxpy(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

void cblas_saxpby(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float beta, float *y, CBLAS_INDEX incy);
void cblas_daxpby(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double beta, double *y, CBLAS_INDEX incy);

void cblas_sswap(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);
void cblas_dswap(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

void cblas_srot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float c, float s);
void cblas_drot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double c, double s);

float cblas_sasum(CBLAS_INDEX n, float *x, CBLAS_INDEX incx);
double cblas_dasum(CBLAS_INDEX n, double *x, CBLAS_INDEX incx);

float cblas_snrm2(CBLAS_INDEX n, float *x, CBLAS_INDEX incx);
double cblas_dnrm2(CBLAS_INDEX n, double *x, CBLAS_INDEX incx);

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
void cblas_sgemm();
void cblas_dgemm();

//------------------------------------------------------
// Utility functions
//------------------------------------------------------
void cblas_init();
void cblas_shutdown();
void cblas_set_num_threads(int threads);
int cblas_get_num_threads(void);
void cblas_init_server();

void cblas_execute(int items, work_queue_t* queue);
void cblas_execute_async(int items, work_queue_t* queue);
void cblas_execute_async_join(int items, work_queue_t* queue);

void cblas_level1_exec(CBLAS_INDEX stride, kernel_function kernel, CBLAS_INDEX n, void* x, CBLAS_INDEX incx, const void* y, CBLAS_INDEX incy);

int cpu_get_core_count();
const char *cpu_get_core_name();

#endif
