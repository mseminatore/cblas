//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
// single-precision vector dot product kernel
//------------------------------------------------------
static void cblas_sdot_k(cblas_args_t* args)
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
// single-precision vector dot product kernel inc=1
//------------------------------------------------------
static void cblas_sdot_k_noinc(cblas_args_t* args)
{
    register float sum = 0.0f;
    float* x = args->x;
    float* y = args->y;
    float* result = args->c;
    register CBLAS_INDEX n = args->n;
    CBLAS_INDEX i = 0;
    register float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (; i + 4 <= n; i += 4)
    {
        sum0 += *x * *y;
        sum1 += *(x + 1) * *(y + 1);
        sum2 += *(x + 2) * *(y + 2);
        sum3 += *(x + 3) * *(y + 3);

        x += 4;
        y += 4;
    }

    sum = sum0 + sum1 + sum2 + sum3;

    for (; i < n; i++)
    {
        sum += *x++ * *y++;
    }

    // set return value
    *result = sum;
}

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
    
    if (mt_used)
    {
        float thread_partial_sums[MAX_THREADS];

        kernel_function kernel = cblas_sdot_k;

        // special case kernel for no increments
        if (incx == 1 && incy == 1)
            kernel = cblas_sdot_k_noinc;

        cblas_level1_exec_result(sizeof(float), kernel, n, x, incx, y, incy, thread_partial_sums);

        // accumulate results
        CBLAS_INDEX threads = cblas_get_num_threads();
        for (CBLAS_INDEX i = 0; i < threads; i++)
        {
            sum += thread_partial_sums[i];
        }
    }
    else
    {
        if (incx == 1 && incy == 1)
        {
            CBLAS_INDEX i = 0;
            register float sum0, sum1, sum2, sum3 = 0.0f;

            for (; i + 4 <= n; i += 4)
            {
                sum0 = *x * *y;
                sum1 = *(x + 1) * *(y + 1);
                sum2 = *(x + 2) * *(y + 2);
                sum3 = *(x + 3) * *(y + 3);

                x += 4;
                y += 4;

                sum += sum0 + sum1 + sum2 + sum3;
            }

            for (; i < n; i++)
            {
                sum += *x++ * *y++;
            }
        }
        else
        {
            // incx and/or incy are not 1
            for (CBLAS_INDEX i = 0; i < n; i++)
            {
                sum += *x * *y;
                x += incx;
                y += incy;
            }
        }
    }

#else
    int mt_used = 0;
    if (incx == 1 && incy == 1)
    {
        CBLAS_INDEX i = 0;
        register float sum0, sum1, sum2, sum3 = 0.0f;

        for (; i + 4 <= n; i += 4)
        {
            sum0 = *x * *y;
            sum1 = *(x + 1) * *(y + 1);
            sum2 = *(x + 2) * *(y + 2);
            sum3 = *(x + 3) * *(y + 3);

            x += 4;
            y += 4;

            sum += sum0 + sum1 + sum2 + sum3;
        }

        for (; i < n; i++)
        {
            sum += *x++ * *y++;
        }
    }
    else
    {
        // incx and/or incy are not 1
        for (CBLAS_INDEX i = 0; i < n; i++)
        {
            sum += *x * *y;
            x += incx;
            y += incy;
        }
    }
#endif

    CBLAS_STATS_END("sdot", n, mt_used);

    return sum;
}

//------------------------------------------------------
// Level-1 double-precision vector dot product
//------------------------------------------------------
double cblas_ddot(CBLAS_INDEX n, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy)
{
    double sum = 0.0;

#ifdef CBLAS_CHECK_INPUTS

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (n <= 0)
        info = 1;
    else if (!x)
        info = 2;
    else if (!y)
        info = 4;

    if (info) {
        XERBLA(info);
        return sum;
    }
#else
    if (n <= 0 || !x || !y)
    {
        assert(n > 0 && x && y);
        return 0.0;
    }
#endif  // CBLAS_XERBLA_INPUTS
#endif  // CBLAS_CHECK_INPUTS

    CBLAS_STATS_START();

    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        sum += *x * *y;
        x += incx;
        y += incy;
    }

    CBLAS_STATS_END("ddot", n, 0);

    return sum;
}
