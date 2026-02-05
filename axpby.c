//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Base scalar kernel for saxpby (y = alpha*x + beta*y)
//------------------------------------------------------
void cblas_saxpby_k(cblas_args_t *args)
{
    float *x = args->x;
    float *y = args->y;
    float alpha = *(float*)args->alpha;
    float beta = *(float*)args->beta;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *y = alpha * *x + beta * *y;
        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
// Base scalar kernel for saxpby with inc=1
//------------------------------------------------------
void cblas_saxpby_k_noinc(cblas_args_t *args)
{
    float *x = args->x;
    float *y = args->y;
    float alpha = *(float*)args->alpha;
    float beta = *(float*)args->beta;
    CBLAS_INDEX n = args->n;

    CBLAS_INDEX i = 0;
    CBLAS_INDEX n4 = n & ~3;

    // 4-way unrolled loop
    for (; i < n4; i += 4)
    {
        y[i]     = alpha * x[i]     + beta * y[i];
        y[i + 1] = alpha * x[i + 1] + beta * y[i + 1];
        y[i + 2] = alpha * x[i + 2] + beta * y[i + 2];
        y[i + 3] = alpha * x[i + 3] + beta * y[i + 3];
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

//------------------------------------------------------
// Level-1 single-precision y = alpha * x + beta * y
//------------------------------------------------------
void cblas_saxpby(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float beta, float *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_AXPBY(n, alpha, x, incx, beta, y, incy, );

    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_AXPY) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.saxpby_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.saxpby_k_noinc;
    }

    if (mt_used)
    {
        cblas_level1_exec(sizeof(float), kernel, n, x, incx, y, incy, &alpha, &beta, "SAXPBY");
    }
    else
    {
        cblas_args_t args = {
            .n = n,
            .incx = incx,
            .incy = incy,
            .x = x,
            .y = y,
            .alpha = &alpha,
            .beta = &beta
        };

        kernel(&args);
    }

    CBLAS_STATS_END("saxpby", n, mt_used);
}

//------------------------------------------------------
// Base scalar kernel for daxpby (y = alpha*x + beta*y)
//------------------------------------------------------
void cblas_daxpby_k(cblas_args_t *args)
{
    double *x = args->x;
    double *y = args->y;
    double alpha = *(double*)args->alpha;
    double beta = *(double*)args->beta;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX incx = args->incx;
    CBLAS_INDEX incy = args->incy;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *y = alpha * *x + beta * *y;
        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
// Base scalar kernel for daxpby with inc=1
//------------------------------------------------------
void cblas_daxpby_k_noinc(cblas_args_t *args)
{
    double *x = args->x;
    double *y = args->y;
    double alpha = *(double*)args->alpha;
    double beta = *(double*)args->beta;
    CBLAS_INDEX n = args->n;

    CBLAS_INDEX i = 0;
    CBLAS_INDEX n4 = n & ~3;

    // 4-way unrolled loop
    for (; i < n4; i += 4)
    {
        y[i]     = alpha * x[i]     + beta * y[i];
        y[i + 1] = alpha * x[i + 1] + beta * y[i + 1];
        y[i + 2] = alpha * x[i + 2] + beta * y[i + 2];
        y[i + 3] = alpha * x[i + 3] + beta * y[i + 3];
    }

    // Handle remaining elements
    for (; i < n; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

//------------------------------------------------------
// Level-1 double-precision y = alpha * x + beta * y
//------------------------------------------------------
void cblas_daxpby(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double beta, double *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_AXPBY(n, alpha, x, incx, beta, y, incy, );

    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_AXPY) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.daxpby_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.daxpby_k_noinc;
    }

    if (mt_used)
    {
        cblas_level1_exec(sizeof(double), kernel, n, x, incx, y, incy, &alpha, &beta, "DAXPBY");
    }
    else
    {
        cblas_args_t args = {
            .n = n,
            .incx = incx,
            .incy = incy,
            .x = x,
            .y = y,
            .alpha = &alpha,
            .beta = &beta
        };

        kernel(&args);
    }

    CBLAS_STATS_END("daxpby", n, mt_used);
}
