//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

//------------------------------------------------------
// single-precision vector dot product kernel
//------------------------------------------------------
void cblas_sdot_k(cblas_args_t* args)
{
    float sum = 0.0f;
    float* x = args->x;
    float* y = args->y;
    float* result = args->c;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *y;
        x += incx;
        y += incy;
    }

    // set return value
    *result = sum;
}

//------------------------------------------------------
// single-precision vector dot product kernel inc==1
//------------------------------------------------------
void cblas_sdot_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    float* y = args->y;
    float* result = args->c;
    register CBLAS_INDEX n = args->n;

    // Scalar implementation with 4-way unrolling
    CBLAS_INDEX i = 0;
    register float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    for (; i + 4 <= n; i += 4)
    {
        // Prefetch ahead for next iteration if vector is large
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(x + CBLAS_PREFETCH_DISTANCE, 0, 0);
            CBLAS_PREFETCH(y + CBLAS_PREFETCH_DISTANCE, 0, 0);
        }

        sum0 += *x * *y;
        sum1 += *(x + 1) * *(y + 1);
        sum2 += *(x + 2) * *(y + 2);
        sum3 += *(x + 3) * *(y + 3);

        x += 4;
        y += 4;
    }

    register float sum = sum0 + sum1 + sum2 + sum3;

    for (; i < n; i++)
    {
        sum += *x++ * *y++;
    }

    // set return value
    *result = sum;
}

//------------------------------------------------------
// double-precision vector dot product kernel
//------------------------------------------------------
void cblas_ddot_k(cblas_args_t* args)
{
    double sum = 0.0;
    double* x = args->x;
    double* y = args->y;
    double* result = args->c;
    register CBLAS_INDEX incx = args->incx, incy = args->incy, n = args->n;

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *y;
        x += incx;
        y += incy;
    }

    // set return value
    *result = sum;
}

//------------------------------------------------------
// double-precision vector dot product kernel inc=1
//------------------------------------------------------
void cblas_ddot_k_noinc(cblas_args_t* args)
{
    double* x = args->x;
    double* y = args->y;
    double* result = args->c;
    register CBLAS_INDEX n = args->n;

    // Scalar implementation with 4-way unrolling
    CBLAS_INDEX i = 0;
    register double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    int use_prefetch = (n > CBLAS_PREFETCH_THRESHOLD);

    for (; i + 4 <= n; i += 4)
    {
        // Prefetch ahead for next iteration if vector is large
        if (use_prefetch && i + CBLAS_PREFETCH_DISTANCE < n) {
            CBLAS_PREFETCH(x + CBLAS_PREFETCH_DISTANCE, 0, 0);
            CBLAS_PREFETCH(y + CBLAS_PREFETCH_DISTANCE, 0, 0);
        }

        sum0 += *x * *y;
        sum1 += *(x + 1) * *(y + 1);
        sum2 += *(x + 2) * *(y + 2);
        sum3 += *(x + 3) * *(y + 3);

        x += 4;
        y += 4;
    }

    register double sum = sum0 + sum1 + sum2 + sum3;

    for (; i < n; i++)
    {
        sum += *x++ * *y++;
    }

    // set return value
    *result = sum;
}
