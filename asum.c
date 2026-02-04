//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

//------------------------------------------------------
// single-precision asum kernel
//------------------------------------------------------
void cblas_sasum_k(cblas_args_t* args)
{
    float* x = args->x;
	register CBLAS_INDEX incx = args->incx, n = args->n;
    float* result = args->c;
    float sum = 0.0f;

    // incx != 1
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += fabsf(*x);
        x += incx;
    }

    *result = sum;
}

//------------------------------------------------------
// single-precision asum kernel incx == 1
//------------------------------------------------------
void cblas_sasum_k_noinc(cblas_args_t* args)
{
    float* x = args->x;
    CBLAS_INDEX n = args->n;
    float* result = args->c;
    float sum = 0.0f;

    // scalar implementation with unrolling
    CBLAS_INDEX i = 0;
    register float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (; i + 4 <= n; i += 4)
    {
        sum0 += fabsf(x[i]);
        sum1 += fabsf(x[i + 1]);
        sum2 += fabsf(x[i + 2]);
        sum3 += fabsf(x[i + 3]);
    }

    sum = sum0 + sum1 + sum2 + sum3;

    for (; i < n; i++)
    {
        sum += fabsf(x[i]);
    }

    *result = sum;
}

//------------------------------------------------------
// Level-1 single-precision vector sum
//------------------------------------------------------
float cblas_sasum(CBLAS_INDEX n, float *x, CBLAS_INDEX incx)
{
    float sum = 0.0f;

    CBLAS_VALIDATE_VEC1(n, x, incx, sum);
    CBLAS_STATS_START();

    int mt_used = 0;

    kernel_function kernel = blas_kernels.sasum_k;

    if (incx == 1)
    {
        kernel = blas_kernels.sasum_k_noinc;
    }

    if (mt_used)
    {
        float thread_partial_sums[MAX_THREADS];
        cblas_level1_exec_result(sizeof(float), kernel, n, x, incx, NULL, 0, thread_partial_sums, "SASUM");
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
        args.c = &sum;
        kernel(&args);
	}

    CBLAS_STATS_END("sasum", n, mt_used);

    return sum;
}

//------------------------------------------------------
// double-precision asum kernel
//------------------------------------------------------
void cblas_dasum_k(cblas_args_t* args)
{
    double* x = args->x;
    register CBLAS_INDEX incx = args->incx, n = args->n;
    double* result = args->c;
    double sum = 0.0;

    // incx != 1
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += fabs(*x);
        x += incx;
    }

    *result = sum;
}

//------------------------------------------------------
// double-precision asum kernel incx == 1
//------------------------------------------------------
void cblas_dasum_k_noinc(cblas_args_t* args)
{
    double* x = args->x;
    CBLAS_INDEX n = args->n;
    double* result = args->c;
    double sum = 0.0;

    // scalar implementation with unrolling
    CBLAS_INDEX i = 0;
    register double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    for (; i + 4 <= n; i += 4)
    {
        sum0 += fabs(x[i]);
        sum1 += fabs(x[i + 1]);
        sum2 += fabs(x[i + 2]);
        sum3 += fabs(x[i + 3]);
    }

    sum = sum0 + sum1 + sum2 + sum3;

    for (; i < n; i++)
    {
        sum += fabs(x[i]);
    }

    *result = sum;
}
//------------------------------------------------------
// Level-1 double-precision vector sum
//------------------------------------------------------
double cblas_dasum(CBLAS_INDEX n, double *x, CBLAS_INDEX incx)
{
    double sum = 0.0;

    CBLAS_VALIDATE_VEC1(n, x, incx, sum);
    CBLAS_STATS_START();

    int mt_used = 0;

    kernel_function kernel = blas_kernels.dasum_k;

    if (incx == 1)
    {
        kernel = blas_kernels.dasum_k_noinc;
    }

    if (mt_used)
    {
        double thread_partial_sums[MAX_THREADS];
        cblas_level1_exec_result(sizeof(double), kernel, n, x, incx, NULL, 0, thread_partial_sums, "DASUM");
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
        args.c = &sum;
        kernel(&args);
    }

    CBLAS_STATS_END("dasum", n, mt_used);

    return sum;
}
