//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#ifndef __CBLAS_H
#define __CBLAS_H

#include <stddef.h>
#include <math.h>

#ifndef MAX_THREADS
#   define MAX_THREADS 8
#endif

//------------------------------------------------------
//
//------------------------------------------------------
typedef size_t CBLAS_INDEX;

typedef int CBLAS_LAYOUT;

enum {
    CblasRowMajor,
    CblasColMajor
};

//------------------------------------------------------
// Level 1 functions
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
// Level 2 functions
//------------------------------------------------------
void cblas_sger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *a, CBLAS_INDEX lda);
void cblas_dger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double *a, CBLAS_INDEX lda);

void cblas_sgemv();
void cblas_dgemv();

//------------------------------------------------------
// Level 3 functions
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

#endif
