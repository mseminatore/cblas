//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

//------------------------------------------------------
// single-precision nrm2 kernel
//------------------------------------------------------
void cblas_snrm2_k(cblas_args_t* args)
{
    float* x = args->x;
    float* result = args->c;
    register CBLAS_INDEX incx = args->incx, n = args->n;
    float sum = 0.0f;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *x;
        x += incx;
    }

    *result = sqrtf(sum);
}

//------------------------------------------------------
// single-precision nrm2 kernel inc==1
//------------------------------------------------------
void cblas_snrm2_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    float* result = args->c;
    register CBLAS_INDEX n = args->n;

    // Scalar implementation with 4-way unrolling
    CBLAS_INDEX i = 0;
    register float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (; i + 4 <= n; i += 4)
    {
        sum0 += x[i] * x[i];
        sum1 += x[i + 1] * x[i + 1];
        sum2 += x[i + 2] * x[i + 2];
        sum3 += x[i + 3] * x[i + 3];
    }

    register float sum = sum0 + sum1 + sum2 + sum3;

    for (; i < n; i++)
    {
        sum += x[i] * x[i];
    }

    *result = sqrtf(sum);
}

//------------------------------------------------------
// double-precision nrm2 kernel
//------------------------------------------------------
void cblas_dnrm2_k(cblas_args_t* args)
{
    double* x = args->x;
    double* result = args->c;
    register CBLAS_INDEX incx = args->incx, n = args->n;
    double sum = 0.0;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *x;
        x += incx;
    }

    *result = sqrt(sum);
}

//------------------------------------------------------
// double-precision nrm2 kernel inc=1
//------------------------------------------------------
void cblas_dnrm2_k_noinc(cblas_args_t* args)
{
    double* x = args->x;
    double* result = args->c;
    register CBLAS_INDEX n = args->n;

    // Scalar implementation with 4-way unrolling
    CBLAS_INDEX i = 0;
    register double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    for (; i + 4 <= n; i += 4)
    {
        sum0 += x[i] * x[i];
        sum1 += x[i + 1] * x[i + 1];
        sum2 += x[i + 2] * x[i + 2];
        sum3 += x[i + 3] * x[i + 3];
    }

    register double sum = sum0 + sum1 + sum2 + sum3;

    for (; i < n; i++)
    {
        sum += x[i] * x[i];
    }

    *result = sqrt(sum);
}
