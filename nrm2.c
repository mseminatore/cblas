//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Level-1 single-precision euclidean norm
//------------------------------------------------------
float cblas_snrm2(CBLAS_INDEX n, float *x, CBLAS_INDEX incx)
{
    float sum = 0.0f;

    CBLAS_VALIDATE_VEC1(n, x, incx, sum);
    CBLAS_STATS_START();

#if defined(MT_ENABLED)
    int mt_used = (n > CBLAS_MT_DOT) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.snrm2_k;

    // special case kernel for no increments
    if (incx == 1)
    {
        kernel = blas_kernels.snrm2_k_noinc;
    }

    if (mt_used)
    {
        float thread_partial_sums[MAX_THREADS];

        cblas_level1_exec_result(sizeof(float), kernel, n, x, incx, NULL, 0, thread_partial_sums, "SNRM2");

        // accumulate partial sums of squares
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
        args.c = &sum;

        kernel(&args);
    }

    CBLAS_STATS_END("snrm2", n, mt_used);

    return sum;
}

//------------------------------------------------------
// Level-1 double-precision euclidean norm
//------------------------------------------------------
double cblas_dnrm2(CBLAS_INDEX n, double *x, CBLAS_INDEX incx)
{
    double sum = 0.0;

    CBLAS_VALIDATE_VEC1(n, x, incx, sum);
    CBLAS_STATS_START();

#if defined(MT_ENABLED)
    int mt_used = (n > CBLAS_MT_DOT) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.dnrm2_k;

    // special case kernel for no increments
    if (incx == 1)
    {
        kernel = blas_kernels.dnrm2_k_noinc;
    }

    if (mt_used)
    {
        double thread_partial_sums[MAX_THREADS];

        cblas_level1_exec_result(sizeof(double), kernel, n, x, incx, NULL, 0, thread_partial_sums, "DNRM2");

        // accumulate partial sums of squares
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
        args.c = &sum;

        kernel(&args);
    }

    CBLAS_STATS_END("dnrm2", n, mt_used);

    return sum;
}

