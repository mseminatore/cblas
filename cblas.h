#ifndef __CBLAS_H
#define __CBLAS_H

#include <stddef.h>

typedef size_t CBLAS_INDEX;

// Level 1 functions
float cblas_sdot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, const float *y, CBLAS_INDEX incy);
double cblas_ddot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, const double *y, CBLAS_INDEX incy);

void cblas_scopy(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, const float *y, CBLAS_INDEX incy);
void cblas_dcopy(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, const double *y, CBLAS_INDEX incy);

void cblas_sscal(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx);
void cblas_dscal(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx);

void cblas_saxpy(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);
void cblas_daxpy(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

void cblas_sswap(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);
void cblas_dswap(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

// Level 2 functions

// Level 3 functions

// Utility functions
void cblas_set_num_threads(int threads);
int cblas_get_num_threads(void);

#endif
