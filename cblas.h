#ifndef __CBLAS_H
#define __CBLAS_H

#include <stddef.h>

typedef size_t CBLAS_INDEX;

// Level 1
float cblas_sdot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);
double cblas_ddot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

void cblas_saxpy();
void cblas_daxpy();

void cblas_sswap(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy);
void cblas_dswap(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy);

void cblas_sscal(CBLAS_INDEX n, const float a, float *y, CBLAS_INDEX incx);
void cblas_dscal(CBLAS_INDEX n, const double a, double *y, CBLAS_INDEX incx);

// Level 2

// Level 3

// Utility
void cblas_set_num_threads(int threads);
int cblas_get_num_threads(void);

#endif
