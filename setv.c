//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// single-precision setv kernel inc == 1
//------------------------------------------------------
void cblas_ssetv_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    float v = *(float*)args->alpha;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        x[i] = v;
        x[i + 1] = v;
        x[i + 2] = v;
        x[i + 3] = v;
    }

    for (; i < n; i++)
        x[i] = v;
}

//------------------------------------------------------
// single-precision setv kernel
//------------------------------------------------------
void cblas_ssetv_k(cblas_args_t* args)
{
    float* x = args->x;
    float v = *(float*)args->alpha;
    register CBLAS_INDEX incx = args->incx, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = v;
        x += incx;
    }
}

//------------------------------------------------------
// double-precision setv kernel inc == 1
//------------------------------------------------------
void cblas_dsetv_k_noinc(cblas_args_t* args)
{
    double* x = args->x;
    double v = *(double*)args->alpha;
    register CBLAS_INDEX n = args->n;
    register CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        x[i] = v;
        x[i + 1] = v;
        x[i + 2] = v;
        x[i + 3] = v;
    }

    for (; i < n; i++)
        x[i] = v;
}

//------------------------------------------------------
// double-precision setv kernel
//------------------------------------------------------
void cblas_dsetv_k(cblas_args_t* args)
{
    double* x = args->x;
    double v = *(double*)args->alpha;
    register CBLAS_INDEX incx = args->incx, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = v;
        x += incx;
    }
}

//------------------------------------------------------
// single precision set vector
//------------------------------------------------------
void cblas_ssetv(CBLAS_INDEX n, float *x, float v)
{
    CBLAS_VALIDATE_VEC1(n, x, 1, );
    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.ssetv_k_noinc;

    if (mt_used)
    {
        cblas_level1_exec(sizeof(float), kernel, n, x, 1, NULL, 1, &v, NULL, "SSETV");
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = 1;
        args.alpha = &v;

        kernel(&args);
    }

    CBLAS_STATS_END("ssetv", n, mt_used);
}

//------------------------------------------------------
// double precision set vector
//------------------------------------------------------
void cblas_dsetv(CBLAS_INDEX n, double *x, double v)
{
    CBLAS_VALIDATE_VEC1(n, x, 1, );
    CBLAS_STATS_START();

#ifdef MT_ENABLED
    int mt_used = (n > CBLAS_MT_COPY) ? 1 : 0;
#else
    int mt_used = 0;
#endif

    kernel_function kernel = blas_kernels.dsetv_k_noinc;

    if (mt_used)
    {
        cblas_level1_exec(sizeof(double), kernel, n, x, 1, NULL, 1, &v, NULL, "DSETV");
    }
    else
    {
        cblas_args_t args;
        args.n = n;
        args.x = x;
        args.incx = 1;
        args.alpha = &v;

        kernel(&args);
    }

    CBLAS_STATS_END("dsetv", n, mt_used);
}
