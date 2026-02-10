//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

//------------------------------------------------------
// single-precision copy kernel incx == incy == 1
//------------------------------------------------------
void cblas_scopy_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        y[i] = x[i];
        y[i + 1] = x[i + 1];
        y[i + 2] = x[i + 2];
        y[i + 3] = x[i + 3];
    }

    for (; i < n; i++)
        y[i] = x[i];
}

//------------------------------------------------------
// single-precision copy kernel
//------------------------------------------------------
void cblas_scopy_k(cblas_args_t *args)
{
    float *x = args->x;
    float *y = args->y;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;
    register CBLAS_INDEX i = 0;

    for (; i < n; i++)
    {
        *y = *x;
        x += incx;
        y += incy;
    }
}

//------------------------------------------------------
// double-precision copy kernel incx == incy == 1
//------------------------------------------------------
void cblas_dcopy_k_noinc(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        y[i] = x[i];
        y[i + 1] = x[i + 1];
        y[i + 2] = x[i + 2];
        y[i + 3] = x[i + 3];
    }

    for (; i < n; i++)
        y[i] = x[i];
}

//------------------------------------------------------
// double-precision copy kernel
//------------------------------------------------------
void cblas_dcopy_k(cblas_args_t* args)
{
    double *x = args->x;
    double *y = args->y;
    register CBLAS_INDEX i = 0;

    for (; i < args->n; i++)
    {
        *y = *x;
        x += args->incx;
        y += args->incy;
    }
}

//------------------------------------------------------
// Level-1 single-precision copy
//------------------------------------------------------
void cblas_scopy(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, );
    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY && cblas_get_num_threads() > 1) ? 1 : 0;
#else
	int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.scopy_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.scopy_k_noinc;
    }

    if (mt_used)
    {
        cblas_level1_exec(sizeof(float), kernel, n, x, incx, y, incy, NULL, NULL, "SCOPY");
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

    CBLAS_STATS_END("scopy", n, mt_used);
}

//------------------------------------------------------
// Level-1 double-precision copy
//------------------------------------------------------
void cblas_dcopy(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, );
    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY && cblas_get_num_threads() > 1) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.dcopy_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.dcopy_k_noinc;
    }

    if (mt_used)
    {
        cblas_level1_exec(sizeof(double), kernel, n, x, incx, y, incy, NULL, NULL, "DCOPY");
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

    CBLAS_STATS_END("dcopy", n, mt_used);
}
