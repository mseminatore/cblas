//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// Level-1 single-precision vector scale
//------------------------------------------------------
void cblas_sscal(CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx)
{
    CBLAS_VALIDATE_SCAL(n, alpha, x, incx, );
    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    if (alpha == 1.0f)
    {
        // nothing to do!!
        return;
    }

    kernel_function kernel = blas_kernels.sscal_k;

    // special case kernel for no increments
    if (incx == 1)
    {
        kernel = blas_kernels.sscal_k_noinc;
    }

    if (mt_used)
    {
        cblas_level1_exec(sizeof(float), kernel, n, x, incx, NULL, 1, &alpha, NULL, "SSCAL");
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = incx;
        args.alpha = &alpha;

        kernel(&args);
    }

    CBLAS_STATS_END("sscal", n, mt_used);
}

//------------------------------------------------------
// Level-1 double-precision vector scale
//------------------------------------------------------
void cblas_dscal(CBLAS_INDEX n, double alpha, double* x, CBLAS_INDEX incx)
{
#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 3;
    else if (incx <= 0)
        info = 4;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (n <= 0 || !x || incx <= 0)
    {
        assert(n >= 0 && x && incx > 0);
        return;
    }
#endif
#endif

    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    if (alpha == 1.0)
    {
        // nothing to do!!
        return;
    }

    kernel_function kernel = blas_kernels.dscal_k;

    // special case kernel for no increments
    if (incx == 1)
    {
        kernel = blas_kernels.dscal_k_noinc;
    }

    if (mt_used)
    {
        cblas_level1_exec(sizeof(double), kernel, n, x, incx, NULL, 1, &alpha, NULL, "DSCAL");
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = incx;
        args.alpha = &alpha;

        kernel(&args);
    }

    CBLAS_STATS_END("dscal", n, mt_used);
}

