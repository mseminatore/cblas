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

    int mt_used = 0;

    kernel_function kernel = blas_kernels.snrm2_k;

    // special case kernel for no increments
    if (incx == 1)
    {
        kernel = blas_kernels.snrm2_k_noinc;
    }

    cblas_args_t args;
    args.n = n;
    args.x = x;
    args.incx = incx;
    args.c = &sum;

    kernel(&args);

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

    int mt_used = 0;

    kernel_function kernel = blas_kernels.dnrm2_k;

    // special case kernel for no increments
    if (incx == 1)
    {
        kernel = blas_kernels.dnrm2_k_noinc;
    }

    cblas_args_t args;
    args.n = n;
    args.x = x;
    args.incx = incx;
    args.c = &sum;

    kernel(&args);

    CBLAS_STATS_END("dnrm2", n, mt_used);

    return sum;
}

