//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

//------------------------------------------------------
// single-precision swap kernel incx == incy == 1
//------------------------------------------------------
void cblas_sswap_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        float t0 = y[i];
        float t1 = y[i + 1];
        float t2 = y[i + 2];
        float t3 = y[i + 3];

        y[i] = x[i];
        y[i + 1] = x[i + 1];
        y[i + 2] = x[i + 2];
        y[i + 3] = x[i + 3];

        x[i] = t0;
        x[i + 1] = t1;
        x[i + 2] = t2;
        x[i + 3] = t3;
    }

    for (; i < n; i++)
    {
        float temp = y[i];
        y[i] = x[i];
        x[i] = temp;
    }
}

//------------------------------------------------------
// single-precision swap kernel
//------------------------------------------------------
void cblas_sswap_k(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        float temp = *y;
        *y = *x;
        *x = temp;

        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
// double-precision swap kernel incx == incy == 1
//------------------------------------------------------
void cblas_dswap_k_noinc(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        double t0 = y[i];
        double t1 = y[i + 1];
        double t2 = y[i + 2];
        double t3 = y[i + 3];

        y[i] = x[i];
        y[i + 1] = x[i + 1];
        y[i + 2] = x[i + 2];
        y[i + 3] = x[i + 3];

        x[i] = t0;
        x[i + 1] = t1;
        x[i + 2] = t2;
        x[i + 3] = t3;
    }

    for (; i < n; i++)
    {
        double temp = y[i];
        y[i] = x[i];
        x[i] = temp;
    }
}

//------------------------------------------------------
// double-precision swap kernel
//------------------------------------------------------
void cblas_dswap_k(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        double temp = *y;
        *y = *x;
        *x = temp;

        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
// Level-1 single-precision swap
//------------------------------------------------------
void cblas_sswap(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, );
    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.sswap_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.sswap_k_noinc;
    }

    if (mt_used)
    {
        cblas_level1_exec(sizeof(float), kernel, n, x, incx, y, incy, NULL, NULL, "SSWAP");
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = incx;
        args.y = y;
        args.incy = incy;

        kernel(&args);
    }

    CBLAS_STATS_END("sswap", n, mt_used);
}

//------------------------------------------------------
// Level-1 double-precision swap
//------------------------------------------------------
void cblas_dswap(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, );
    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.dswap_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.dswap_k_noinc;
    }

    if (mt_used)
    {
        cblas_level1_exec(sizeof(double), kernel, n, x, incx, y, incy, NULL, NULL, "DSWAP");
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = incx;
        args.y = y;
        args.incy = incy;

        kernel(&args);
    }

    CBLAS_STATS_END("dswap", n, mt_used);
}
