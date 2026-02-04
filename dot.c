//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

//------------------------------------------------------
// Level-1 single-precision vector dot product
//------------------------------------------------------
float cblas_sdot(CBLAS_INDEX n, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy)
{
    float sum = 0.0f;

    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, sum);
    CBLAS_STATS_START();

#if defined(MT_ENABLED)
    int mt_used = (n > CBLAS_MT_DOT) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.sdot_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.sdot_k_noinc;
    }

    if (mt_used)
    {
        float thread_partial_sums[MAX_THREADS];

        cblas_level1_exec_result(sizeof(float), kernel, n, x, incx, y, incy, thread_partial_sums, "SDOT");

        // accumulate results
        CBLAS_INDEX threads = cblas_get_num_threads();
        for (CBLAS_INDEX i = 0; i < threads; i++)
        {
            sum += thread_partial_sums[i];
        }
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = incx;
        args.y = y;
        args.incy = incy;
        args.c = &sum;

        kernel(&args);
    }

    CBLAS_STATS_END("sdot", n, mt_used);

    return sum;
}

//------------------------------------------------------
// Level-1 double-precision vector dot product
//------------------------------------------------------
double cblas_ddot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    double sum = 0.0;

    CBLAS_VALIDATE_VEC2(n, x, incx, y, incy, sum);
    CBLAS_STATS_START();

#if defined(MT_ENABLED)
    int mt_used = (n > CBLAS_MT_DOT) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.ddot_k;

    // special case kernel for no increments
    if (incx == 1 && incy == 1)
    {
        kernel = blas_kernels.ddot_k_noinc;
    }

    if (mt_used)
    {
        double thread_partial_sums[MAX_THREADS];

        cblas_level1_exec_result(sizeof(double), kernel, n, x, incx, y, incy, thread_partial_sums, "DDOT");

        // accumulate results
        CBLAS_INDEX threads = cblas_get_num_threads();
        for (CBLAS_INDEX i = 0; i < threads; i++)
        {
            sum += thread_partial_sums[i];
        }
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = incx;
        args.y = y;
        args.incy = incy;
        args.c = &sum;

        kernel(&args);
    }

    CBLAS_STATS_END("ddot", n, mt_used);

    return sum;
}
