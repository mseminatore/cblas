//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// single-precision scal kernel
//------------------------------------------------------
void cblas_sscal_k(cblas_args_t* args)
{
    float* x = args->x;
    float alpha = *(float*)args->alpha;
    register CBLAS_INDEX incx = args->incx, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = alpha * *x;
        x += incx;
    }
}

//------------------------------------------------------
// single-precision scal kernel inc==1
//------------------------------------------------------
void cblas_sscal_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    float alpha = *(float*)args->alpha;
    register CBLAS_INDEX n = args->n;

    // Scalar implementation with 4-way unrolling
    CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        x[i] = alpha * x[i];
        x[i + 1] = alpha * x[i + 1];
        x[i + 2] = alpha * x[i + 2];
        x[i + 3] = alpha * x[i + 3];
    }

    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}

//------------------------------------------------------
// double-precision scal kernel
//------------------------------------------------------
void cblas_dscal_k(cblas_args_t* args)
{
    double* x = args->x;
    double alpha = *(double*)args->alpha;
    register CBLAS_INDEX incx = args->incx, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        *x = alpha * *x;
        x += incx;
    }
}

//------------------------------------------------------
// double-precision scal kernel inc=1
//------------------------------------------------------
void cblas_dscal_k_noinc(cblas_args_t* args)
{
    double* x = args->x;
    double alpha = *(double*)args->alpha;
    register CBLAS_INDEX n = args->n;

    // Scalar implementation with 4-way unrolling
    CBLAS_INDEX i = 0;

    for (; i + 4 <= n; i += 4)
    {
        x[i] = alpha * x[i];
        x[i + 1] = alpha * x[i + 1];
        x[i + 2] = alpha * x[i + 2];
        x[i + 3] = alpha * x[i + 3];
    }

    for (; i < n; i++)
    {
        x[i] = alpha * x[i];
    }
}
