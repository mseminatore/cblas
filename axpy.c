//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#ifdef _WIN32
#	include <malloc.h>
#else
#	include <alloca.h>
#endif

//------------------------------------------------------
// single-precision axpy kernel incx == incy == 1
//------------------------------------------------------
void cblas_saxpy_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    float alpha = *(float*)args->alpha;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        y[i] = alpha * x[i] + y[i];
        y[i + 1] = alpha * x[i + 1] + y[i + 1];
        y[i + 2] = alpha * x[i + 2] + y[i + 2];
        y[i + 3] = alpha * x[i + 3] + y[i + 3];
    }

    for (; i < n; i++)
        y[i] = alpha * x[i] + y[i];
}

//------------------------------------------------------
// single-precision axpy kernel (arbitrary increments)
//------------------------------------------------------
void cblas_saxpy_k(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    float alpha = *(float*)args->alpha;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *y = alpha * *x + *y;
        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
// double-precision axpy kernel incx == incy == 1
//------------------------------------------------------
void cblas_daxpy_k_noinc(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    double alpha = *(double*)args->alpha;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        y[i] = alpha * x[i] + y[i];
        y[i + 1] = alpha * x[i + 1] + y[i + 1];
        y[i + 2] = alpha * x[i + 2] + y[i + 2];
        y[i + 3] = alpha * x[i + 3] + y[i + 3];
    }

    for (; i < n; i++)
        y[i] = alpha * x[i] + y[i];
}

//------------------------------------------------------
// double-precision axpy kernel (arbitrary increments)
//------------------------------------------------------
void cblas_daxpy_k(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    double alpha = *(double*)args->alpha;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *y = alpha * *x + *y;
        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
// Level-1 single-precision y = alpha * x + y
//------------------------------------------------------
void cblas_saxpy(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_AXPY(n, alpha, x, incx, y, incy, );
    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_AXPY) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.saxpy_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.saxpy_k_noinc;
    }

    if (mt_used)
    {
        cblas_level1_exec(sizeof(float), kernel, n, x, incx, y, incy, &alpha, NULL, "SAXPY");
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = incx;
        args.y = y;
        args.incy = incy;
        args.alpha = &alpha;

        kernel(&args);
    }

    CBLAS_STATS_END("saxpy", n, mt_used);
}

//------------------------------------------------------
// Level-1 double-precision y = alpha * x + y
//------------------------------------------------------
void cblas_daxpy(CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_AXPY(n, alpha, x, incx, y, incy, );
    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_AXPY) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.daxpy_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.daxpy_k_noinc;
    }

    if (mt_used)
    {
        cblas_level1_exec(sizeof(double), kernel, n, x, incx, y, incy, &alpha, NULL, "DAXPY");
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = incx;
        args.y = y;
        args.incy = incy;
        args.alpha = &alpha;

        kernel(&args);
    }

    CBLAS_STATS_END("daxpy", n, mt_used);
}
